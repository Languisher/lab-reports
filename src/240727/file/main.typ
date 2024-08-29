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
  title: [FaaSFlow: Enable Efficient Workflow Execution for Function-as-a-Service],
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
  image("../figures/image.png", width: 80%)
)

- ASPLos '22, February 28 - March 4, 2022
- Keywords: _Decentralization_
]

#outline-slide()

= Review

== Serverless Workflow

#slide[
#tblock(title: "Serverless Workflow")[
  Serverless functions are event-driven, and they need to be executed in a pre-defined order. Such a diagram with nodes connected by edges in a DAG form is known as the *serverless workflow*.
]

#figure(
  image("../figures/workflow.png", width: 50%)
)
]

#slide[
#tblock(title: "Control-Plane and Data-Plane")[
  - *Control-Plane*: User-defined execution order 
  - *Data-Plane*: Runtime data dependency

  Usually identical and static. However, in serverless context, auto-scaling and warm containers may lead to multiple and different scales in the data-plane.
]
]


= Motivation

== MasterSP and its Limitations

#slide[
#tblock(title: "Master-side Workflow Schedule Pattern")[
*Centralized* Workflow: Central workflow engine in the master node determines whether a function task is triggered to run or not.

]

- Engine makes resource provision
- Task $T_f$ triggered only if its predecessors are all completed:
  1. Assign $T_f$ from the master engine
  2. Execute $T_f$ invocation
  3. Return the exectuion state to the master engine

][
#figure(
  image("../figures/workersp.png", width: 80%)
)
#figure(
  image("../figures/workersp2.png", width: 85%)
)
]

#slide[

Problems:
  - #underline[Large scheduling overhead]: Transfer of function execution states
  - #underline[Large data movement overhead]: Additional database storage services for temp data storage and delivery

#figure(
  image("../figures/workersp-latency.png", width:55%)
)
]

== Data-Shipping Pattern

#slide[
#tblock(title: "Data-Shipping Pattern")[
  Each time a function task runs, the input data needs to be fetched from its predecessor functions, then read into memory for execution by the container executor. Such process is called a *data-shipping pattern*.
]

Problems:
- Function isolation #underline[brings more overhead of task-to-task data communcation]
- Compulsory for user to use remote storage services
- #underline[Data locality] is not utilized
]


= Architecture

== WorkerSP's Structure Organization
#slide[
The inverse of MasterSP: *Worker-side workflow schedule pattern (WorkerSP)*, _Decentralize_

#tblock(title: "Structure Organization of WorkerSP")[
- Master node scheduling $-->_"Offload"$ Per-worker engine assigned to perform local function triggering and invoking
- Master node only partition a workflow graph into sub-graph (See later)
- _Workflow_ structure introduced with _State_, _FunctionInfo_ and _InvocationID_.
  - _State_: Execution state of functions and their predecessors for invocation synchronization
  - _FunctionInfo_: Meta information for local functions
  - _InvocationID_: Unique state identification
]]

== WorkerSP's State Synchronization

#slide[
(Reminder) Engine of each worker node maintains functions' and their predecessors' execution state in the *local sub-graph*.
#tblock(title:"Example: Invocation Synchronization")[
1. $F_A$ is invoked
2. State pass to Node $B$ and $C$ 
3. $F_B$ and $F_C$ update info 
4. When $F_A$ finished, `PredecessorsDone` of $B$ and $C$ + 1
5. When `PredecessorsDone` = `PredecessorsCount`, local engine of $B$ and $C$ will trigger
]
][
  #figure(
    image("../figures/invo.png")
  )
]

== Overview of FaaSFlow


#slide[
  #tblock(title: "FaaSFlow: Workflow System")[
Three components:
1. *Workflow graph scheduler*
2. *Per-worker workflow engine*
3. Adaptive Storage Library *FaaStore*

  ]

][
#figure(
  image("../figures/faas.png")
)
]

== Component 1: Graph Scheduler

#slide[
1. *DAG Parser* parse the hierarchy _Workflow Definition Language (WDL)_ (which defines a serverless workflow)
#figure(
  image("../figures/logicflows.png")
)

]

#slide[
2. *Graph Partitionning*: Partitioning of DAG

To alleviate the gap between Control-plane and (dynamic data-plane):

- $overline("Scale"(v_i))$: Avg. number of scaled instances of a function node $v_i$ during iteration 
- $overline("Map"(v_i))$: Mapped instances in the data-plane (e.g. `Foreach`)

Partition iteration activated when significant performance degredation
][
#figure(
  image("../figures/algo.png")
)
]


== Component 2: Per-Worker Workflow Engine

#slide[
Maintaining states for different functions.

Direct state communication via *inter-node TCP* or *inner RPC connections*.

#tblock(title: "Red-Black Deployment")[Manage different versions of sub-graph versions in worker engines, only the up-to-date version is getting triggered.]

][

 #figure(
  image("../figures/redblack.png")
) 
]


== Component 3: FaaStore

#slide[
In-memory storage enables data and files reside in local sub-graph; defUlt remote store save them by user configs.

#tblock(title: "In-Memory Quota")[
Well-organized #underline[quota for data movement]

Due to _over-provisionning_, $ O(v_i) = max("Mem"(v_i) - S - mu, 0) $ and $ "Quota"(G(V, E)) = sum_(v in V)O(v) $]][
#figure(
  image("../figures/memrealloc.png")
)

]

= Evaluation

== Evaluation

FaaSFlow reduces the scheduling overhead from 712ms to 141.9ms for scientific workflows, and from 181.3ms to 51.4ms for real-world applications on average. All applications can achieve an average of 74.6% scheduling overhead optimization in FaaSFlow.

_Basically did not mention memory allocation improvements ......_