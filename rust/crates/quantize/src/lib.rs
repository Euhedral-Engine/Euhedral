//! The command line of the quantization tools: `quantization COMMAND [FLAGS]`.
//!
//! This crate parses the command and its flags and starts the work; the work is in
//! `euhedral-inference-quantization`. [`COMMANDS`] lists every command the tools have. A command whose work is not
//! wired yet says so and exits with [`EXIT_NOT_AVAILABLE`]; three commands are not part of this tool at all
//! ([`Availability::Dropped`]).

mod package;

/// The process exit code of a run that finished.
pub const EXIT_OK: u8 = 0;
/// The command was valid and its work failed.
pub const EXIT_RUN_FAILED: u8 = 1;
/// The command or its flags are not valid.
pub const EXIT_USAGE: u8 = 2;
/// The command is known but not wired in this build.
pub const EXIT_NOT_AVAILABLE: u8 = 3;

/// What a command is in this tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    /// The command runs.
    Wired,
    /// The command is part of the tool and its work is not wired yet.
    Pending,
    /// The command is not part of this tool: it was an experiment that was measured and not adopted.
    Dropped,
}

/// One command of the tool.
#[derive(Debug, Clone, Copy)]
pub struct Command {
    pub name: &'static str,
    pub availability: Availability,
    /// The flags of the command, in the usage line.
    pub usage: &'static str,
}

/// Every command, in the order the usage lists them.
pub const COMMANDS: [Command; 13] = [
    Command {
        name: "calibrate",
        availability: Availability::Pending,
        usage: "--model CHECKPOINT (--quantization nvfp4|q3 | --format trellis --bits 2.5|3.0|3.5) --tokens TOKENS.npy --draft-ids-from ARTIFACT --out DIR",
    },
    Command {
        name: "score",
        availability: Availability::Pending,
        usage: "--artifact ARTIFACT --results RESULTS.txt (--reference REFERENCE.kld --name NAME)...",
    },
    Command {
        name: "generate",
        availability: Availability::Pending,
        usage: "--artifact ARTIFACT --tokenizer CHECKPOINT --out CORPUS_DIR (--prompts PROMPTS.jsonl)...",
    },
    Command {
        name: "train-drafter",
        availability: Availability::Pending,
        usage: "--corpus CORPUS_DIR --model CHECKPOINT --artifact ARTIFACT --out DIR",
    },
    Command {
        name: "export-drafter",
        availability: Availability::Pending,
        usage: "--model CHECKPOINT --artifact BASE_ARTIFACT --out ARTIFACT",
    },
    Command {
        name: "train-blocks",
        availability: Availability::Dropped,
        usage: "",
    },
    Command {
        name: "train-model",
        availability: Availability::Dropped,
        usage: "",
    },
    Command {
        name: "rotate",
        availability: Availability::Dropped,
        usage: "",
    },
    Command {
        name: "assemble",
        availability: Availability::Pending,
        usage: "--allocation FILE --source OPTION=ARTIFACT... --base OPTION --out ARTIFACT",
    },
    Command {
        name: "cost",
        availability: Availability::Pending,
        usage: "--source OPTION=ARTIFACT... --base OPTION --out COSTS.json",
    },
    Command {
        name: "allocate",
        availability: Availability::Pending,
        usage: "--source OPTION=ARTIFACT... --base OPTION --costs COSTS.json --priors PRIORS.json --reference R.kld... --context N --state SEARCH.resume --out DIR",
    },
    Command {
        name: "reference",
        availability: Availability::Pending,
        usage: "--artifact ARTIFACT --tokenizer CHECKPOINT --text FILE --out OUT.kld",
    },
    Command {
        name: "package-tokenizer",
        availability: Availability::Wired,
        usage: "--artifact ARTIFACT --tokenizer CHECKPOINT [--out ARTIFACT]",
    },
];

/// The command called `name`.
pub fn find(name: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|command| command.name == name)
}

/// The usage text: one line per command that is part of the tool.
pub fn usage() -> String {
    let mut text = String::from("usage: quantization COMMAND [FLAGS]\n");
    for command in COMMANDS.iter().filter(|c| c.availability != Availability::Dropped) {
        text.push_str(&format!("       {} {}\n", command.name, command.usage));
    }
    text
}

/// What a run says and how it ends.
#[derive(Debug, PartialEq, Eq)]
pub struct Outcome {
    pub code: u8,
    /// The message for standard error; empty when there is none.
    pub message: String,
}

/// Runs `args`, the command line without the program name.
pub fn run(args: &[String]) -> Outcome {
    let Some(name) = args.first() else {
        return Outcome {
            code: EXIT_USAGE,
            message: usage(),
        };
    };
    let Some(command) = find(name) else {
        return Outcome {
            code: EXIT_USAGE,
            message: format!("unknown command '{name}'\n{}", usage()),
        };
    };
    match command.availability {
        Availability::Wired => match command.name {
            "package-tokenizer" => package::run(&args[1..]),
            _ => Outcome {
                code: EXIT_NOT_AVAILABLE,
                message: format!("the command '{name}' has no work behind it\n"),
            },
        },
        Availability::Pending => Outcome {
            code: EXIT_NOT_AVAILABLE,
            message: format!("the command '{name}' is not available in this build yet\n"),
        },
        Availability::Dropped => Outcome {
            code: EXIT_NOT_AVAILABLE,
            message: format!("the command '{name}' is not part of this tool\n"),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn run_words(words: &[&str]) -> Outcome {
        run(&words.iter().map(|word| word.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn the_tool_has_thirteen_distinct_commands() {
        let mut names: Vec<&str> = COMMANDS.iter().map(|command| command.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 13);
    }

    #[test]
    fn no_command_and_an_unknown_command_print_the_usage() {
        let none = run_words(&[]);
        assert_eq!(none.code, EXIT_USAGE);
        assert!(none.message.contains("calibrate --model"));
        let unknown = run_words(&["frobnicate"]);
        assert_eq!(unknown.code, EXIT_USAGE);
        assert!(unknown.message.starts_with("unknown command 'frobnicate'"));
    }

    #[test]
    fn a_pending_command_says_it_is_not_available() {
        let outcome = run_words(&["calibrate", "--model", "x"]);
        assert_eq!(outcome.code, EXIT_NOT_AVAILABLE);
        assert!(outcome.message.contains("not available"));
    }

    #[test]
    #[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
    fn the_flags_of_package_tokenizer_are_checked_before_anything_runs() {
        let words = |words: &[&str]| words.iter().map(|word| word.to_string()).collect::<Vec<_>>();
        let parsed = package::parse(&words(&["--artifact", "a", "--tokenizer=c", "--out", "o"])).unwrap();
        assert_eq!(parsed.out.as_deref(), Some(Path::new("o")));
        assert!(package::parse(&words(&["--artifact", "a"])).is_err());
        assert!(package::parse(&words(&["--artifact", "a", "--artifact", "b", "--tokenizer", "c"])).is_err());
        assert!(package::parse(&words(&["--artifact", "a", "--tokenizer", "c", "--bogus"])).is_err());
        assert!(package::parse(&words(&["--artifact"])).is_err());
        assert_eq!(run_words(&["package-tokenizer"]).code, EXIT_USAGE);
    }

    #[test]
    fn a_dropped_command_is_known_but_not_listed() {
        for name in ["train-blocks", "train-model", "rotate"] {
            assert_eq!(run_words(&[name]).code, EXIT_NOT_AVAILABLE);
            assert!(!usage().contains(name));
        }
    }
}
