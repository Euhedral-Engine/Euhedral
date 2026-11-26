//! One artifact file, opened through `AsyncIo`: its header, metadata and table (and, when asked, its tokenizer
//! section) are read and checked; the payloads stay in the file, which stays open.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use euhedral_inference_edrl::{
    EdrlError, FOOTER_BYTES, HEADER_BYTES, Header, Metadata, ModelMetadata, ReadPlan, Region, TensorTable,
    TokenizerFooter, TokenizerSection, content_end,
};
use euhedral_io::{AsyncIo, OwnedFile};

use super::{AssembleError, Done};
use crate::checkpoint::io::{Next, file_length, open_read, read_exact};

/// An open artifact file and what its regions say.
#[derive(Debug)]
pub struct ArtifactFile {
    path: PathBuf,
    file: Arc<OwnedFile>,
    length: u64,
    metadata: ModelMetadata,
    table: TensorTable,
    tokenizer: Option<TokenizerSection>,
}

impl ArtifactFile {
    /// Opens the artifact at `path` for reading and checks its header, metadata and table. With `tokenizer` its
    /// tokenizer section is read too (it has one or it does not: see [`ArtifactFile::tokenizer`]).
    pub fn open(
        io: &Arc<AsyncIo>,
        path: &Path,
        tokenizer: bool,
        done: Done<ArtifactFile>,
    ) -> Result<(), AssembleError> {
        let io = Arc::clone(io);
        let path = path.to_path_buf();
        let next_io = Arc::clone(&io);
        let next_path = path.clone();
        open_read(
            &io,
            &path,
            Next::new(move |opened| match opened {
                Err(error) => done(Err(error.into())),
                Ok(file) => {
                    let opening = Opening {
                        io: next_io,
                        path: next_path,
                        file: Arc::new(file),
                        tokenizer,
                        length: 0,
                        header: None,
                        footer: None,
                        metadata: None,
                        table: None,
                        done,
                    };
                    opening.measure();
                }
            }),
        )?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The open file, for reading payloads.
    pub fn file(&self) -> &Arc<OwnedFile> {
        &self.file
    }

    /// The file's length.
    pub fn length(&self) -> u64 {
        self.length
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    pub fn table(&self) -> &TensorTable {
        &self.table
    }

    /// The tokenizer section, when it was asked for and the file has one.
    pub fn tokenizer(&self) -> Option<&TokenizerSection> {
        self.tokenizer.as_ref()
    }
}

/// The steps that read an artifact's regions, one after the other.
struct Opening {
    io: Arc<AsyncIo>,
    path: PathBuf,
    file: Arc<OwnedFile>,
    tokenizer: bool,
    length: u64,
    header: Option<Header>,
    footer: Option<TokenizerFooter>,
    metadata: Option<ModelMetadata>,
    table: Option<TensorTable>,
    done: Done<ArtifactFile>,
}

impl Opening {
    fn artifact_error(&self, source: EdrlError) -> AssembleError {
        AssembleError::Artifact {
            path: self.path.display().to_string(),
            source,
        }
    }

    fn fail(self, error: AssembleError) {
        (self.done)(Err(error));
    }

    /// Asks for the file's length, then the header.
    fn measure(self) {
        let report = Arc::clone(&self.done);
        let file = Arc::clone(&self.file);
        let io = Arc::clone(&self.io);
        let context = self.path.display().to_string();
        let next = Next::<u64>::new(move |length| match length {
            Err(error) => self.fail(error.into()),
            Ok(length) => self.with_length(length),
        });
        if let Err(error) = file_length(&io, file, context, next) {
            report(Err(error.into()));
        }
    }

    fn with_length(mut self, length: u64) {
        self.length = length;
        match ReadPlan::probe(length) {
            Err(error) => {
                let error = self.artifact_error(error);
                self.fail(error);
            }
            Ok(_) => self.read(0..HEADER_BYTES as u64, "the header", Opening::with_header),
        }
    }

    fn with_header(mut self, bytes: Vec<u8>) {
        let parsed = <[u8; HEADER_BYTES]>::try_from(bytes.as_slice())
            .map_err(|_| EdrlError::Truncated { what: "the header" })
            .and_then(|bytes| Header::parse(&bytes));
        match parsed {
            Err(error) => {
                let error = self.artifact_error(error);
                self.fail(error);
            }
            Ok(header) => {
                self.header = Some(header);
                let footer_at = self.length.saturating_sub(FOOTER_BYTES as u64);
                let range = footer_at..self.length;
                self.read(range, "the footer", Opening::with_footer);
            }
        }
    }

    fn with_footer(mut self, bytes: Vec<u8>) {
        let parsed = <[u8; FOOTER_BYTES]>::try_from(bytes.as_slice())
            .map_err(|_| EdrlError::Truncated { what: "the footer" })
            .and_then(|bytes| TokenizerFooter::parse(&bytes));
        let planned = parsed.and_then(|footer| {
            let header = self
                .header
                .as_ref()
                .ok_or(EdrlError::Truncated { what: "the header" })?;
            let plan = ReadPlan::new(header, self.length, footer.as_ref())?;
            Ok((footer, plan))
        });
        match planned {
            Err(error) => {
                let error = self.artifact_error(error);
                self.fail(error);
            }
            Ok((footer, plan)) => {
                self.footer = footer;
                match plan.range(Region::Metadata) {
                    Some(range) => self.read(range, "the metadata", Opening::with_metadata),
                    None => self.fail(AssembleError::invalid("the read plan has no metadata")),
                }
            }
        }
    }

    fn with_metadata(mut self, bytes: Vec<u8>) {
        match Metadata::parse(&bytes) {
            Err(error) => {
                let error = self.artifact_error(error);
                self.fail(error);
            }
            Ok(metadata) => {
                self.metadata = Some(metadata);
                let range = self.header.as_ref().map(Header::table_range);
                match range {
                    Some(range) => self.read(range, "the tensor table", Opening::with_table),
                    None => self.fail(AssembleError::invalid("the header is missing")),
                }
            }
        }
    }

    fn with_table(mut self, bytes: Vec<u8>) {
        let parsed = match self.header.as_ref() {
            Some(header) => {
                content_end(self.length, self.footer.as_ref()).and_then(|end| TensorTable::parse(&bytes, header, end))
            }
            None => Err(EdrlError::Truncated { what: "the header" }),
        };
        match parsed {
            Err(error) => {
                let error = self.artifact_error(error);
                self.fail(error);
            }
            Ok(table) => {
                self.table = Some(table);
                match self
                    .footer
                    .as_ref()
                    .filter(|_| self.tokenizer)
                    .map(|f| f.body_range(self.length))
                {
                    None => self.finish(None),
                    Some(Err(error)) => {
                        let error = self.artifact_error(error);
                        self.fail(error);
                    }
                    Some(Ok(range)) => self.read(range, "the tokenizer section", Opening::with_tokenizer),
                }
            }
        }
    }

    fn with_tokenizer(self, body: Vec<u8>) {
        let parsed = match self.footer.as_ref() {
            Some(footer) => TokenizerSection::parse(body, footer),
            None => Err(EdrlError::Truncated { what: "the footer" }),
        };
        match parsed {
            Err(error) => {
                let error = self.artifact_error(error);
                self.fail(error);
            }
            Ok(section) => self.finish(Some(section)),
        }
    }

    fn finish(self, tokenizer: Option<TokenizerSection>) {
        let Opening {
            path,
            file,
            length,
            metadata,
            table,
            done,
            ..
        } = self;
        match (metadata, table) {
            (Some(metadata), Some(table)) => done(Ok(ArtifactFile {
                path,
                file,
                length,
                metadata,
                table,
                tokenizer,
            })),
            _ => done(Err(AssembleError::invalid("the artifact's regions were not all read"))),
        }
    }

    /// Reads `range` of the file and goes on with `then`; an empty range goes on with no bytes.
    fn read(self, range: Range<u64>, what: &'static str, then: fn(Opening, Vec<u8>)) {
        let Ok(length) = usize::try_from(range.end - range.start) else {
            let message = format!("{}: {what} does not fit in memory", self.path.display());
            self.fail(AssembleError::invalid(message));
            return;
        };
        if length == 0 {
            then(self, Vec::new());
            return;
        }
        let report = Arc::clone(&self.done);
        let io = Arc::clone(&self.io);
        let file = Arc::clone(&self.file);
        let context = format!("{what} of {}", self.path.display());
        let next = Next::<Vec<u8>>::new(move |bytes| match bytes {
            Err(error) => self.fail(error.into()),
            Ok(bytes) => then(self, bytes),
        });
        if let Err(error) = read_exact(&io, file, range.start, length, context, next) {
            report(Err(error.into()));
        }
    }
}
