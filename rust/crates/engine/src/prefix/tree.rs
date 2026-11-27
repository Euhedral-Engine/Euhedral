//! A radix tree of checkpoints over token ids.
//!
//! A node covers the tokens between its parent's position and its own; a lookup walks down the nodes whose tokens
//! equal the prompt's, comparing the tokens themselves, so a match is exact. Nodes become visible when published, and
//! are evicted least recently used first, childless nodes only, never while pinned. Positions lie on the KV page grid,
//! so a node's pages start where its parent's end. A node owns the slots that hold its bytes: evicting or aborting it
//! returns them. Not thread-safe: confined to its owner.

use std::sync::Arc;

use euhedral_memory::{SlotArena, SlotLease};

use crate::residency::footprint::PAGE_TOKENS;

/// Every node position is a multiple of this: the number of tokens in a KV page.
pub const POSITION_GRANULE: u32 = PAGE_TOKENS as u32;

/// The kind of speculative decoding state a node holds beside the base state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speculation {
    /// The MTP layer: its cache's pages and the base hidden row of the last position.
    Mtp,
    /// The DFlash2 drafter: each layer's ring of keys and values of the last window of positions.
    Dflash2,
}

/// A node of one tree. Ids of evicted or aborted nodes are never reused for another node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId {
    index: u32,
    generation: u32,
}

/// Why a reservation was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ReserveError {
    /// The span does not advance, overruns the tokens or leaves the page grid.
    #[error("a node must advance, on the page grid, within the tokens")]
    Span,
    /// The parent was evicted or aborted.
    #[error("the parent node is gone")]
    Stale,
    /// Not enough slots, and nothing evictable remains.
    #[error("no slots free and nothing evictable")]
    NoRoom,
}

/// One stored checkpoint: the sequence state at `position`, whose parent holds the state at `start`.
#[derive(Debug)]
pub struct PrefixNode {
    parent: Option<NodeId>,
    start: u32,
    position: u32,
    tokens: Vec<u32>,
    /// The token at `position` of the prompt that stored the node, when it had one: the token the speculative state's
    /// last row was paired with.
    paired: Option<u32>,
    speculation: Option<Speculation>,
    slots: Vec<SlotLease>,
    children: Vec<NodeId>,
    pins: u32,
    last_use: u64,
    published: bool,
}

impl PrefixNode {
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    pub fn start(&self) -> u32 {
        self.start
    }

    pub fn position(&self) -> u32 {
        self.position
    }

    /// The tokens at positions `[start, position)`.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The token the node's last speculative row was paired with: the prompt's token at `position` when the node was
    /// stored. A restore into a prompt with the same token there can use that row as it is.
    pub fn paired(&self) -> Option<u32> {
        self.paired
    }

    /// The kind of speculative state the node holds beside the base state. It is usable only with an unbroken chain of
    /// ancestors of the same kind.
    pub fn speculation(&self) -> Option<Speculation> {
        self.speculation
    }

    /// The slots that hold the node's bytes, in layout order.
    pub fn slots(&self) -> &[SlotLease] {
        &self.slots
    }

    /// Whether a lookup's match (or an open reservation under it) holds the node.
    pub fn pinned(&self) -> bool {
        self.pins > 0
    }

    fn matches(&self, prompt: &[u32]) -> bool {
        prompt.get(self.start as usize..self.position as usize) == Some(self.tokens.as_slice())
    }
}

/// The nodes a lookup matched, from the first under the root to the deepest. Pinned until released.
#[derive(Debug)]
#[must_use]
pub struct Match {
    chain: Vec<NodeId>,
    leaf: NodeId,
}

impl Match {
    pub fn chain(&self) -> &[NodeId] {
        &self.chain
    }

    /// The deepest node: the state the match restores.
    pub fn leaf(&self) -> NodeId {
        self.leaf
    }
}

#[derive(Debug)]
struct Entry {
    generation: u32,
    node: Option<PrefixNode>,
}

#[derive(Debug)]
pub struct PrefixTree {
    arena: Arc<SlotArena>,
    entries: Vec<Entry>,
    vacant: Vec<u32>,
    published: Vec<NodeId>,
    root: NodeId,
    clock: u64,
    evictions: u64,
}

impl PrefixTree {
    pub fn new(arena: Arc<SlotArena>) -> Self {
        let root = PrefixNode {
            parent: None,
            start: 0,
            position: 0,
            tokens: Vec::new(),
            paired: None,
            speculation: None,
            slots: Vec::new(),
            children: Vec::new(),
            pins: 0,
            last_use: 0,
            published: true,
        };
        PrefixTree {
            arena,
            entries: vec![Entry {
                generation: 0,
                node: Some(root),
            }],
            vacant: Vec::new(),
            published: Vec::new(),
            root: NodeId {
                index: 0,
                generation: 0,
            },
            clock: 0,
            evictions: 0,
        }
    }

    pub fn root(&self) -> NodeId {
        self.root
    }

    /// The node `id` names, or `None` once it was evicted or aborted.
    pub fn get(&self, id: NodeId) -> Option<&PrefixNode> {
        let entry = self.entries.get(id.index as usize)?;
        if entry.generation == id.generation {
            entry.node.as_ref()
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(crate) fn node(&self, id: NodeId) -> &PrefixNode {
        self.get(id).unwrap()
    }

    fn get_mut(&mut self, id: NodeId) -> Option<&mut PrefixNode> {
        let entry = self.entries.get_mut(id.index as usize)?;
        if entry.generation == id.generation {
            entry.node.as_mut()
        } else {
            None
        }
    }

    fn position_of(&self, id: NodeId) -> Option<u32> {
        self.get(id).map(PrefixNode::position)
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// The longest match of `prompt` that leaves at least one token to prefill. With a `speculation` kind only nodes
    /// that hold that kind of speculative state match, so the chain is unbroken; with `None` any node matches.
    pub fn lookup(&mut self, prompt: &[u32], speculation: Option<Speculation>) -> Option<Match> {
        let mut chain = Vec::new();
        let mut at = self.root;
        loop {
            let mut best: Option<(NodeId, u32)> = None;
            let children = self.get(at).map_or(&[][..], |node| node.children.as_slice());
            for &child_id in children {
                let Some(child) = self.get(child_id) else {
                    continue;
                };
                if child.position as usize >= prompt.len() {
                    continue;
                }
                if speculation.is_some() && speculation != child.speculation {
                    continue;
                }
                if best.is_some_and(|(_, reach)| child.position <= reach) {
                    continue;
                }
                if child.matches(prompt) {
                    best = Some((child_id, child.position));
                }
            }
            let Some((best, _)) = best else { break };
            chain.push(best);
            at = best;
        }
        let leaf = *chain.last()?;
        for &id in &chain {
            let stamp = self.tick();
            if let Some(node) = self.get_mut(id) {
                node.pins += 1;
                node.last_use = stamp;
            }
        }
        Some(Match { chain, leaf })
    }

    pub fn release(&mut self, matched: Match) {
        for id in matched.chain {
            if let Some(node) = self.get_mut(id) {
                node.pins = node.pins.saturating_sub(1);
            }
        }
    }

    /// The published child of `parent` whose span `[parent.position, position)` equals that of `tokens` and whose
    /// speculative state is of the kind `speculation` (`None`: none). A span can be stored once per kind.
    pub fn find(
        &mut self,
        parent: NodeId,
        tokens: &[u32],
        position: u32,
        speculation: Option<Speculation>,
    ) -> Option<NodeId> {
        self.find_where(parent, tokens, position, |child| child.speculation == speculation)
    }

    /// The published child of `parent` whose span equals that of `tokens`, whatever speculative state it holds: every
    /// node holds the base state a plain capture would store.
    pub fn find_any(&mut self, parent: NodeId, tokens: &[u32], position: u32) -> Option<NodeId> {
        self.find_where(parent, tokens, position, |_| true)
    }

    fn find_where(
        &mut self,
        parent: NodeId,
        tokens: &[u32],
        position: u32,
        accept: impl Fn(&PrefixNode) -> bool,
    ) -> Option<NodeId> {
        if position as usize > tokens.len() {
            return None;
        }
        let found = self.get(parent)?.children.iter().copied().find(|&id| {
            self.get(id)
                .is_some_and(|child| child.position == position && accept(child) && child.matches(tokens))
        })?;
        let stamp = self.tick();
        self.get_mut(found)?.last_use = stamp;
        Some(found)
    }

    /// Leases `slots` slots for a node covering `[parent.position, position)` of `tokens`, evicting least recently
    /// used nodes for room. The node is invisible until [`publish`](Self::publish), and `parent` stays pinned until
    /// [`publish`](Self::publish) or [`abort`](Self::abort).
    pub fn reserve(
        &mut self,
        parent: NodeId,
        tokens: &[u32],
        position: u32,
        speculation: Option<Speculation>,
        slots: usize,
    ) -> Result<NodeId, ReserveError> {
        let start = self.position_of(parent).ok_or(ReserveError::Stale)?;
        if position <= start || position as usize > tokens.len() || !position.is_multiple_of(POSITION_GRANULE) {
            return Err(ReserveError::Span);
        }
        if slots > self.arena.capacity() as usize {
            return Err(ReserveError::NoRoom);
        }
        self.unpin_or_pin(parent, true);
        let mut leased = Vec::with_capacity(slots);
        while leased.len() < slots {
            if let Some(lease) = self.arena.lease() {
                leased.push(lease);
                continue;
            }
            let Some(victim) = self.victim() else {
                self.unpin_or_pin(parent, false);
                return Err(ReserveError::NoRoom);
            };
            self.evict(victim);
        }
        let speculation = self.effective_speculation(parent, speculation);
        let node = PrefixNode {
            parent: Some(parent),
            start,
            position,
            tokens: tokens[start as usize..position as usize].to_vec(),
            paired: tokens.get(position as usize).copied(),
            speculation,
            slots: leased,
            children: Vec::new(),
            pins: 0,
            last_use: 0,
            published: false,
        };
        Ok(self.insert(node))
    }

    /// The kind a node reserved under `parent` with `requested` really holds: the request when `parent` is the root or
    /// holds the same kind, else none (a speculative node under a base-only parent is unusable). Size a reservation by
    /// this, not by the request.
    pub fn effective_speculation(&self, parent: NodeId, requested: Option<Speculation>) -> Option<Speculation> {
        match self.get(parent) {
            Some(node) if parent == self.root || node.speculation == requested => requested,
            _ => None,
        }
    }

    fn unpin_or_pin(&mut self, id: NodeId, pin: bool) {
        if let Some(node) = self.get_mut(id) {
            node.pins = if pin {
                node.pins + 1
            } else {
                node.pins.saturating_sub(1)
            };
        }
    }

    fn insert(&mut self, node: PrefixNode) -> NodeId {
        match self.vacant.pop() {
            Some(index) => {
                let entry = &mut self.entries[index as usize];
                entry.node = Some(node);
                NodeId {
                    index,
                    generation: entry.generation,
                }
            }
            None => {
                self.entries.push(Entry {
                    generation: 0,
                    node: Some(node),
                });
                NodeId {
                    index: (self.entries.len() - 1) as u32,
                    generation: 0,
                }
            }
        }
    }

    fn remove(&mut self, id: NodeId) -> Option<PrefixNode> {
        let entry = self.entries.get_mut(id.index as usize)?;
        if entry.generation != id.generation {
            return None;
        }
        let node = entry.node.take()?;
        entry.generation = entry.generation.wrapping_add(1);
        self.vacant.push(id.index);
        Some(node)
    }

    /// Makes a reserved node visible to lookups. Does nothing for a node already published or gone.
    pub fn publish(&mut self, id: NodeId) {
        let stamp = self.tick();
        let Some(node) = self.get_mut(id) else { return };
        if node.published {
            return;
        }
        node.published = true;
        node.last_use = stamp;
        let Some(parent) = node.parent else { return };
        if let Some(parent) = self.get_mut(parent) {
            parent.children.push(id);
            parent.pins = parent.pins.saturating_sub(1);
        }
        self.published.push(id);
    }

    /// Gives a reserved node's slots back; it never became visible. Refused (`false`, nothing changes) for a node
    /// that is published, gone, or the parent of an open reservation: abort that child first, or a child could be
    /// published under a parent that no longer exists.
    pub fn abort(&mut self, id: NodeId) -> bool {
        if self.get(id).is_none_or(|node| node.published || node.pins > 0) {
            return false;
        }
        let Some(node) = self.remove(id) else {
            return false;
        };
        if let Some(parent) = node.parent {
            self.unpin_or_pin(parent, false);
        }
        true
    }

    /// Published nodes.
    pub fn size(&self) -> usize {
        self.published.len()
    }

    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// The least recently used published node with no children and no pins.
    fn victim(&self) -> Option<NodeId> {
        self.published
            .iter()
            .copied()
            .filter_map(|id| self.get(id).map(|node| (id, node)))
            .filter(|(_, node)| node.children.is_empty() && node.pins == 0)
            .min_by_key(|(_, node)| node.last_use)
            .map(|(id, _)| id)
    }

    fn evict(&mut self, id: NodeId) {
        let Some(node) = self.remove(id) else { return };
        if let Some(parent) = node.parent.and_then(|parent| self.get_mut(parent)) {
            parent.children.retain(|&child| child != id);
        }
        self.published.retain(|&published| published != id);
        self.evictions += 1;
    }
}
