//! A lock-free stack for the retire list and the pinned pool.

use std::fmt;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

struct Node<T> {
    value: T,
    next: *mut Node<T>,
}

/// A Treiber stack that is only ever emptied whole.
///
/// Any thread pushes; [`Stack::take_all`] detaches the entire chain with one swap, so no pop can see a node another
/// thread has already recycled (the ABA hazard of a one-node pop). Nothing blocks and nothing spins on a lock.
pub(crate) struct Stack<T> {
    head: AtomicPtr<Node<T>>,
}

// SAFETY: the stack owns its values and hands each to exactly one thread, so it is as thread-safe as `T`.
unsafe impl<T: Send> Send for Stack<T> {}
// SAFETY: see `Send`; every operation goes through the atomic head.
unsafe impl<T: Send> Sync for Stack<T> {}

impl<T> Stack<T> {
    pub(crate) fn new() -> Stack<T> {
        Stack {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }

    pub(crate) fn push(&self, value: T) {
        let node = Box::into_raw(Box::new(Node {
            value,
            next: ptr::null_mut(),
        }));
        let mut head = self.head.load(Ordering::Relaxed);
        loop {
            // SAFETY: `node` is ours until the exchange below publishes it.
            unsafe { (*node).next = head };
            match self
                .head
                .compare_exchange_weak(head, node, Ordering::Release, Ordering::Relaxed)
            {
                Ok(_) => return,
                Err(current) => head = current,
            }
        }
    }

    /// Detaches and returns every value, most recently pushed first.
    pub(crate) fn take_all(&self) -> Vec<T> {
        let mut node = self.head.swap(ptr::null_mut(), Ordering::Acquire);
        let mut values = Vec::new();
        while !node.is_null() {
            // SAFETY: the swap gave this thread the whole chain; each node came from `Box::into_raw` and is visited
            // once.
            let boxed = unsafe { Box::from_raw(node) };
            node = boxed.next;
            values.push(boxed.value);
        }
        values
    }

    /// One value, or `None`; the rest are pushed back.
    pub(crate) fn pop(&self) -> Option<T> {
        let mut all = self.take_all();
        let one = all.pop();
        for rest in all {
            self.push(rest);
        }
        one
    }
}

impl<T> Drop for Stack<T> {
    fn drop(&mut self) {
        drop(self.take_all());
    }
}

impl<T> fmt::Debug for Stack<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Stack").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_all_returns_every_value_newest_first_and_empties_the_stack() {
        let stack = Stack::new();
        for value in 1..=3 {
            stack.push(value);
        }
        assert_eq!(stack.take_all(), vec![3, 2, 1]);
        assert!(stack.take_all().is_empty());
    }

    #[test]
    fn pop_leaves_the_rest() {
        let stack = Stack::new();
        stack.push("a");
        stack.push("b");
        assert!(stack.pop().is_some());
        assert_eq!(stack.take_all().len(), 1);
        assert_eq!(stack.pop(), None);
    }
}
