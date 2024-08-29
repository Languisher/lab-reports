#import "@preview/cetz:0.2.2"
#import "@preview/fletcher:0.4.5" as fletcher: node, edge
#import "@preview/touying:0.4.2": *
#import "../lib.typ" as buaa-theme

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

// Register university theme
// You can replace it with other themes and it can still work normally
#let s = buaa-theme.register()

// Set the numbering of section and subsection
#let s = (s.methods.numbering)(self: s, section: "1.", "1.1")

// Set the speaker notes configuration, you can show it by pympress
// #let s = (s.methods.show-notes-on-second-screen)(self: s, right)

// Global information configuration
#let s = (s.methods.info)(
  self: s,
  title: [Mooncake: Kimi's KVCache-centric Architecture for LLM Serving],
  subtitle: [(W6) Paper Reading],
  author: [Nan Lin],
  date: datetime.today(),
  institution: [Shanghai Jiao Tong University],
)

// Pdfpc configuration
#let s = (s.methods.append-preamble)(self: s, pdfpc.config(
  duration-minutes: 30,
  start-time: datetime(hour: 14, minute: 00, second: 0),
  end-time: datetime(hour: 14, minute: 30, second: 0),
  last-minutes: 5,
  note-font-size: 12,
  disable-markdown: false,
  default-transition: (
    type: "push",
    duration-seconds: 2,
    angle: ltr,
    alignment: "vertical",
    direction: "inward",
  ),
))

// Extract methods
#let (init, slides, touying-outline, alert, speaker-note, tblock) = utils.methods(s)
#show: init.with(
  lang: "en",
  font: ("Linux Libertine", "Source Han Sans SC", "Source Han Sans"),
)

#show strong: alert

// Extract slide functions
#let (slide, empty-slide, title-slide, outline-slide, new-section-slide, ending-slide) = utils.slides(s)
#show: slides.with()

#slide[
  #figure(
    image("../figures/paper.png")
  )
- Click the link to open: https://arxiv.org/html/2407.00079v1
]

#outline-slide()


= Review

== Review 1: Transformer-based Inference Workflow

#slide[
Generating from an input text sequences (*prompts*) to text (*completion*) consists of:
1. Loading weights to GPU
2. _Tokenizing_ *prompts* on CPU
3. Transfering the tokenized prompts to GPU



][
#figure(
  caption: "Tokenization",
  image("../figures/tokenization.png")
)
]

#slide[
(cont.)
4. *Prefill stage* (or *Initialization stage*): Generate the first token
5. *Decoding stage* (or *Generation stage*, *Auto-regression stage*, *Incremental stage*): Repeat ...
  - Append the generated token to the sequence of input tokens
  - Using it as a new input to generate the second token

  End repeat if EOS token generated or maximum token sequence length reached.
][
#figure(
  caption: "Prefill Stage and Decoding Stage",
  image("../figures/two-phase.png")
)

]

#slide[
  (cont.)
  6. _Detokenizing_ the tokenized incremented token to get your generated text.
][

 #figure(
  caption: "Detokenization",
  image("../figures/detokenization.png")
) 
]

== Review 2: KVCache

#slide[ 
Motivation:
- *Masking* technique: _Causality_, we cannot use token generated of later words to influence the former ones.
#figure(
  caption: "Masking",
  image("../figures/masking.png", width: 60%)
)
- Previous tokens are identical across all the stages.

][
#figure(
  caption: "Excessive Datas in Prefill Stage and Decoding Stage",
  image("../figures/two-phase-kvcache.png")
)
]

#slide[
Changes to the original workflow:
- No longer need tokens but their key and value vectors.
- Take the KVCache and the last generated token as input.
- Initilization phase could be renamed as _pre-fill_ stage since it prepares all the key and value vectors of input text sequence.

Issues of KVCache:
- _Large memory consumption in GPU_
- _Low GPU Utilization problem_: data transmission time > calculation time
][
  #figure(
  caption: "Generating Steps with KVCache Enabled",
  image("../figures/kvcache.png")
)
]



= Motivation

== Modelization


#slide[


#tblock(title:"Primary Goal")[
_Diversified workload_ of LLM models $->$ _Optimization problem_ with multiple constraints:
- Maximize overall throughput
- Constraints of latency-related SLOs: 
]


Workload difference in: 
- I/O length
- Frequency 
- Distribution of arrival
- SLOs

]

== Two Types of SLOs

#slide[

#figure(
  image("../figures/1.png")
)][

Computation time increases:
- At _prefill stage_, superlinearly with input length due to parallel processing of tokens $->$  *Time to first token (TTFT)*
- At _decoding stage_, sublinearly with batch size due to auto-regressive processing on one token at a time per batch $->$ *Time between tokens (TBT)*
]


== Going More Concrete

#slide[

#tblock(title: "Disaggregation Idea")[
Make best use of resources $->$ 
1. Scheduling of KVCache is central to LLM serving scheduling;
2. Decouple and reconstruct nodes for different but collabrative goals
]

Current approaches: 
- Reuse KVCache as much as possible 
- Maximize the number of tokens in each batch (to improve *Model FLOPs Utilization (MFU)*) 

However, corresponding issues:
- If tokens stored in remote places $->$ TTFT problem
- Large batch size $->$ TBT problem
]



== Another Objective
#slide[
Limited GPU/accelerator supply while excessive demand.
- Early reject policy if no available slots could be use 
- However, fluctuation in the workloads

Their work: 

#tblock(title: "Overload-Oriented Scheduling")[
- Prediction of generation length
- Load prediction in the short-tern future
- Classify request priorities.
]
]

= Architecture

== Overview: KVCache-Centric Disaggregated Architecture

#slide[
  #figure( 
    image("../figures/archoverview.png")
  )
][
Quick-reminder: reuse KVCache / max num of tokens in each batch.

#tblock(title:"Conductor: The Global Scheduler")[
Selects a pair of prefill and decoding instances and schedule the request to:
1. Transfer reusable KVCache
2. (Prefill Stage) Complete in chunks/layers, stream the output to the corresponding decoding instance
3. (Decoding Stage) Load KVCache and add it to the batching process, generate output
]
]


== Storage and Transfer Logic of the KVCache Blocks

#slide[
  #figure( 
    image("../figures/2.png")
  )
][
- In CPU memory, KVCache stored as page blocks.
- Cache eviction algorithms used
- Transfer across CPUs and GPUs handled by a RDMA-based component called _Messenger_
- Global scheduler: _Conductor_
]

== Workflow of a Request (C3)

#slide[
  #figure( 
    image("../figures/3.png")
  )
][
After tokenizing is finished, conductor selects a pair of prefill nodes and a decoding node:
1. *KVCache reuse*: loads the prefix cache from remote CPU memory into GPU memory
2. *Incremental Prefill*: Prefill node complete prefill stage using the prefix cache, afterwards it stores the _newly generated and incremental_ KVCache back into CPU memory
]

#slide[
  #figure( 
    image("../figures/3.png")
  )
][
3. *KVCache Transfer*: Operated by _Messenger_, cross-machine KVCache transformer, asynchronously exectued and streaming the KVCache
4. *Decoding*: KVCache is received in the CPU DRAM of the decoding node, request joins the next batch in a continuous batching manner
]

== KVCache-centric Scheduling

#slide[
#tblock(title:"Prefill Global Scheduling")[
Selection of prefill instances depend not only on load, but also _prefix cache hit length_ and _distribution of reusable KVCache blocks_.
]
#figure( 
    image("../figures/s1.png", width:45%)
  )
][
#figure( 
    image("../figures/2.png", width:110%)
  )
]

#slide[
  - Input token for each request divided into blocks, assign hash keys one by one.
  - Compare one by one to identify the prefix match length, named `prefix_len`

  #figure( 
    image("../figures/matched_length.png", width:100%)
  )
]

#slide[
    #figure( 
    image("../figures/alg1.png")
  )
][
- Find shortest TTFT
- Update $T_"cache"$ and $T_"queue"$
- If SLO not achievable, return `HTTP 429 TOO MANY REQUESTS`.
]

#slide[
    - Each prefill machine mangages its own set of prefix caches.
    - Goal: achieve balance between cache matching and instance load (e.g. system prompts vs long document)
    - Collect global usages of each block, use for forecasting ? (No way, requests fluctuation)
  #tblock(title:"Cache Load Balancing")[
    Requests may not always be directed to the prefill instance with the longest prefix cache length due to high instance load.

    _Conductor_ forwards the cache's location and the request to an alternative instance.
  ]

  - Improve replication of hot-spot caches.
]

== Overload-Oriented Scheduling (C6)

#slide[
*Overload*: Unrealistic to process every incoming request.

System should process as many requests as possibles until the system load reaches a predefined threshold.

*Load* = Ratio of num of requests being processed to the system's max capacity

Decision making:
- Whether to accept the prefill stage based on the _prefill instance's load_
- Whether to proceed with the decoding stage based on the _decoding instance's load_

]

#slide[

  *Time-lag* between two instances for a single request:
  If a request is rejected by the decoding instance due to high load after the prefill stage has been completed $->$ Waste of resources

  #tblock(title: "Early Rejection")[
    Advance the load assessement of the decoding instance to precede the beginning of the prefill stage.
  ]
]

#slide[
  Problems of Early Rejection: Fluctuation

     #figure( 
    image("../figures/earlyrejection.png", width: 60%)
  ) 
]

#slide[
  *Early Rejection based on Prediction*

  #tblock(title: "Request level")[
    Predict Output Length, thus estimate accurately TTFT and TBT.
  ]
  #tblock(title: "System level")[
    Estimate the overall batch count or the TBT status for instances after a specified time.
  ]

  Their work: System level, request level left for future work.

]

= Evaluations

== Evaluations

Mooncake demonstrates significantly higher throughput, with enhancements ranging from 50% to 525%, while adhering to the same TTFT and TBT SLO constraints compared to vLLM.

Mooncake can mitigate load fluctuations, increasing the request handling capacity.

= References

== References

#slide[
- Review: LLM Inference
  - https://medium.com/@plienhar/llm-inference-series-2-the-two-phase-process-behind-llms-responses-1ff1ff021cd5
  - https://medium.com/@plienhar/llm-inference-series-3-kv-caching-unveiled-048152e461c8
  - https://www.bilibili.com/video/BV1TZ421j7Ke/?spm_id_from=333.337.search-card.all.click&vd_source=5ade9da381cec8d2c191f450ccd0cf57
]