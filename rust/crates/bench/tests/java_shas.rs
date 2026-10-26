//! The prompts the Java benchmark builds, reproduced byte for byte. The expected counts and digests are the
//! `--validate-only` output of the Java benchmark (the `prompt A/T tokens, sha256 ...` lines: actual and target tokens)
//! for the same artifact's tokenizer, with the default prompt seed. Needs `euhedral.qwen.rco-artifact` (in
//! `~/.gradle/gradle.properties`, or `EUHEDRAL_TEST_QWEN_RCO_ARTIFACT`); reads only the tokenizer section.
//!
//! ```text
//! cargo test -p euhedral-inference-bench -- --ignored
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: a test reads a fixture artifact"
)]
#![allow(
    clippy::print_stdout,
    reason = "a test prints its skip line when the machine has no artifact"
)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::PathBuf;

use euhedral_inference_bench::prompt::words::{DEFAULT_SEED, GENERATOR as WORDS_GENERATOR};
use euhedral_inference_bench::prompt::{PromptCorpus, chat};
use euhedral_inference_edrl::{ReadPlan, Region, TokenizerFooter, TokenizerSection};
use euhedral_inference_tokenizer::Tokenizer;

const KEY: &str = "euhedral.qwen.rco-artifact";

fn read_range(file: &mut File, range: Range<u64>) -> Vec<u8> {
    file.seek(SeekFrom::Start(range.start)).unwrap();
    let mut bytes = vec![0; (range.end - range.start) as usize];
    file.read_exact(&mut bytes).unwrap();
    bytes
}

fn real_tokenizer() -> Option<Tokenizer> {
    let path = PathBuf::from(test_support::require_property(KEY)?);
    if !path.is_file() {
        println!("skipped: {KEY} names no file");
        return None;
    }
    let mut file = File::open(&path).unwrap();
    let file_len = file.metadata().unwrap().len();
    let probe = ReadPlan::probe(file_len).unwrap();
    let footer = TokenizerFooter::parse(
        read_range(&mut file, probe.range(Region::TokenizerFooter).unwrap())
            .as_slice()
            .try_into()
            .unwrap(),
    )
    .unwrap()
    .expect("the artifact carries its tokenizer");
    let body = read_range(&mut file, footer.body_range(file_len).unwrap());
    let section = TokenizerSection::parse(body, &footer).unwrap();
    Some(Tokenizer::from_section(&section).unwrap())
}

/// A target and, for each of its chat prompts in iteration order, the actual token count and the sha256.
type ChatRows = (usize, [(usize, &'static str); 4]);

const CHAT: [ChatRows; 5] = [
    (
        128,
        [
            (120, "d39baa5e85f2f73e0648678dcd7dcb84d81037efbb37e477c3d4d8653e2c4475"),
            (126, "28cece9b489a7e15d0ccb6298b2783c860ff7738a9d1a2ab09c2c23c730a8662"),
            (124, "97ae599f86b4788bd708c0522beb4d432b7d6687cdfbe121a7eab0d35daf2643"),
            (126, "0d64134e8e773568baad55877a9be0f216a91ea02549f0df5eafc6274780048a"),
        ],
    ),
    (
        2000,
        [
            (1987, "8779b9b8d9fa54a34d4c280509c5cd5de231c9d6bcea7d79c2e3682868e31d13"),
            (1993, "d78cea6104a19068ea181c539c5f4f9a1b3fb222c725c870781dcb7d8750fbf1"),
            (1991, "03ed7be7466d38161546a044f30dc53b9836f172242ddea19cf1ea3713353c7b"),
            (1993, "5463168e3cc18ed37a82664563731aeb782e131bc962abe19fd7deeec6e7f219"),
        ],
    ),
    (
        4096,
        [
            (3964, "1b789497efe9b0c467346cb82db1caff5e2074af0f7e9cd0af75a0ac94b1cd07"),
            (3970, "2f7cbfcb3d41e61d61c46b59fb4d32d1d8066531cf05fa22a0187eb02186a3e9"),
            (3968, "d1f50fa701e54d2039fabbdc4ef2594e593908f63890b471c98fbdc21841efa9"),
            (3970, "1451c7a87eda33403fe8809f726281c3715bf4a61edd69f46931849b83504302"),
        ],
    ),
    (
        16384,
        [
            (
                15930,
                "ca3190756ac34301671f4721e93ae1b8b9db4efa8c42e39c9c79f5389599e182",
            ),
            (
                15936,
                "852981f625892b4b64e1239a19461961e01d7bf08f49a58f0260cab593ec8023",
            ),
            (
                15934,
                "de90076c930ded2f8b280fed96a986da640ae01857ca8d088c8a574334b22542",
            ),
            (
                15936,
                "54ed34d27f00448fe13f5b1cd54fb03d4b3cbf7fb5fc6dd584700812f55ebe41",
            ),
        ],
    ),
    (
        32000,
        [
            (
                31906,
                "997e7be7ddabaf9cea9c4fe06a76da017ca48a1102c645e1a4456757580de2ef",
            ),
            (
                31912,
                "c0cc88c6433a121635c0d7beabd643fdf3b59659e5b1ec60f46a9c743a3d6d78",
            ),
            (
                31910,
                "d41703a34840e748fd2b9469ead74b58dd3379c1cfcacfd979b1534df8b91333",
            ),
            (
                31912,
                "b1bfa8256580f26bbb371cf8cb3d0b16662a3527fdde7dca21cc79febd5715fa",
            ),
        ],
    ),
];

/// (target, sha256); every words prompt reaches its target exactly.
const WORDS: [(usize, &str); 4] = [
    (32, "7aec4b00b6b0720c2602425f10587ab1a7ea87818ebb2787b9c10476b6046ad1"),
    (1024, "af41a5786623ff25031f2adc10ca7cf5ac5c28755c96e21c4af95c094e9331c5"),
    (4096, "bbd4557cde4b743f0315732c56bf9a1934c2dad88bb947f3495c3c20c28b02ce"),
    (
        128_000,
        "6103740fae2226534f240d5287056c86e7f15e6b464d4c50caa11aaf61174467",
    ),
];

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn chat_prompts_match_java_sha() {
    let Some(tokenizer) = real_tokenizer() else { return };
    assert_eq!(PromptCorpus::Chat.generator(), "euhedral-chat-v2");
    for (target, expected) in CHAT {
        let prompts = PromptCorpus::Chat.prompts(target, DEFAULT_SEED, &tokenizer).unwrap();
        assert_eq!(prompts.len(), chat::TASKS);
        for (iteration, (prompt, (tokens, sha256))) in prompts.iter().zip(expected).enumerate() {
            assert_eq!(prompt.sha256, sha256, "target {target}, task {iteration}");
            assert_eq!(prompt.ids.len(), tokens, "target {target}, task {iteration}");
        }
        // The iteration picks the task, and wraps.
        let wrapped = PromptCorpus::Chat.prompt(target, 5, DEFAULT_SEED, &tokenizer).unwrap();
        assert_eq!(wrapped, prompts[1]);
    }
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn words_prompts_match_java_sha() {
    let Some(tokenizer) = real_tokenizer() else { return };
    assert_eq!(PromptCorpus::Words.generator(), WORDS_GENERATOR);
    for (target, sha256) in &WORDS[..3] {
        let prompt = PromptCorpus::Words
            .prompt(*target, 0, DEFAULT_SEED, &tokenizer)
            .unwrap();
        assert_eq!(prompt.sha256, *sha256, "target {target}");
        assert_eq!(prompt.ids.len(), *target);
    }
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn words_prompt_128000_matches_java_sha() {
    let Some(tokenizer) = real_tokenizer() else { return };
    let (target, sha256) = WORDS[3];
    let prompt = PromptCorpus::Words.prompt(target, 0, DEFAULT_SEED, &tokenizer).unwrap();
    assert_eq!(prompt.sha256, sha256);
    assert_eq!(prompt.ids.len(), target);
}
