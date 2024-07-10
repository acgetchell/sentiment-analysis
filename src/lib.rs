use std::str::FromStr;

use anyhow::Result;
use spin_sdk::{
    http::{IntoResponse, Json, Params, Request, Response, Router},
    http_component,
    key_value::Store,
    llm::{infer_with_options, InferencingModel::Llama2Chat},
};

use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct SentimentAnalysisRequest {
    pub sentence: String,
}

#[derive(Serialize)]
pub struct SentimentAnalysisResponse {
    pub sentiment: String,
}

const PROMPT: &str = r#"\
<<SYS>>
You are a bot that generates sentiment analysis responses. Respond with a single positive, negative, or neutral.
<</SYS>>
<INST>
Follow the pattern of the following examples:

User: Hi, my name is Bob
Bot: neutral

User: I am so happy today
Bot: positive

User: I am so sad today
Bot: negative
</INST>

User: {SENTENCE}
"#;

/// A Spin HTTP component that internally routes requests.
#[http_component]
fn handle_route(req: Request) -> Response {
    let mut router = Router::new();
    router.any("/api/*", not_found);
    router.post("/api/sentiment-analysis", perform_sentiment_analysis);
    router.handle(req)
}

fn not_found(_: Request, _: Params) -> Result<impl IntoResponse> {
    Ok(Response::new(404, "Not found"))
}

fn perform_sentiment_analysis(req: Request, _params: Params) -> Result<impl IntoResponse> {
    let request = body_json_to_map(&req)?;
    // Do some basic cleanup on the input
    let sentence = request.sentence.trim();
    println!("Performing sentiment analysis on: {}", sentence);

    // Prepare the KV store
    let kv = Store::open_default()?;

    // If the sentiment of the sentence is already in the KV store, return it
    // if kv.exists(sentence).unwrap_or(false) {
    //     println!("Found sentence in KV store returning cached sentiment.");
    //     let sentiment = kv.get(sentence)?;
    //     let resp = SentimentAnalysisResponse {
    //         sentiment: String::from_utf8(sentiment.unwrap())?,
    //     };
    //     let resp_str = serde_json::to_string(&resp)?;
    if let Ok(sentiment) = kv.get(sentence) {
        println!("Found sentence in KV store returning cached sentiment.");
        let resp = SentimentAnalysisResponse {
            sentiment: String::from_utf8(sentiment.unwrap())?,
        };

        // let resp_str = serde_json::to_string(&resp)?;
        // return Ok(Response::new(200, resp_str));
        return send_ok_response(200, resp);
    }
    println!("Sentence not found in KV store.");

    // Perform sentiment analysis
    println!("Running inference...");
    let inferencing_result = infer_with_options(
        Llama2Chat,
        &PROMPT.replace("{SENTENCE}", sentence),
        spin_sdk::llm::InferencingParams {
            max_tokens: 8,
            ..Default::default()
        },
    )?;

    println!("Inference result: {:?}", inferencing_result);

    let sentiment = inferencing_result
        .text
        .lines()
        .next()
        .unwrap_or_default()
        .strip_prefix("Bot:")
        .unwrap_or_default()
        .parse::<Sentiment>();
    println!("Got sentiment: {sentiment:?}");

    if let Ok(sentiment) = sentiment {
        println!("Caching sentiment in KV store.");
        let _ = kv.set(sentence, sentiment.as_str().as_bytes());
    }

    // Cache result in KV store
    let resp = SentimentAnalysisResponse {
        sentiment: sentiment
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default(),
    };

    // let resp_str = serde_json::to_string(&resp)?;
    // Ok(Response::new(200, resp_str))
    send_ok_response(200, resp)
}

// fn send_ok_response(code: u16, resp: SentimentAnalysisResponse) -> Result<Response> {
//     Ok(Response::builder()
//         .status(code)
//         .body(Some(serde_json::to_string(&resp)?.into()))?)
// }

fn send_ok_response(code: u16, resp: SentimentAnalysisResponse) -> Result<Response> {
    let resp_str = serde_json::to_string(&resp)?;
    Ok(Response::new(code, resp_str))
}

fn body_json_to_map(req: &Request) -> Result<SentimentAnalysisRequest> {
    let body = match req.body().as_ref() {
        Some(bytes) => bytes,
        None => anyhow::bail!("Request body is empty"),
    };

    Ok(serde_json::from_slice(&body)?)
}

#[derive(Copy, Clone, Debug)]
enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

impl Sentiment {
    fn as_str(&self) -> &str {
        match self {
            Self::Positive => "positive",
            Self::Negative => "negative",
            Self::Neutral => "neutral",
        }
    }
}

impl std::fmt::Display for Sentiment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<[u8]> for Sentiment {
    fn as_ref(&self) -> &[u8] {
        self.as_str().as_bytes()
    }
}

impl FromStr for Sentiment {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let sentiment = match s.trim() {
            "positive" => Self::Positive,
            "negative" => Self::Negative,
            "neutral" => Self::Neutral,
            _ => return Err(format!("Invalid sentiment: {}", s)),
        };
        Ok(sentiment)
    }
}

// #[http_component]
// fn handle_sentiment_analysis(req: Request) -> anyhow::Result<impl IntoResponse> {
//     println!("Handling request to {:?}", req.header("spin-full-url"));
//     Ok(Response::builder()
//         .status(200)
//         .header("content-type", "text/plain")
//         .body("Hello, Fermyon")
//         .build())
// }
