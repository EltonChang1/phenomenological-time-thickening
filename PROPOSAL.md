# Phenomenological Time-Thickening Test

**Study status** · Draft protocol for preregistration

## Research question

Does a real interval without new sensory or task input change an AI system’s response to an identical one-token probe? If the response changes, can the shift be explained by autonomous state updates, access to a clock, or changes in the serving environment? Does the change affect how the system represents a recent event or predicts an imminent one?

Husserl’s account of internal time-consciousness describes the perception of a temporally extended event, such as a melody, in terms of the current impression, retention of what has just elapsed, and protention toward what is about to occur. William James used *specious present* for the immediate experience of a span of time. This proposal draws on both ideas. Its operational analogues are the influence of the recent past and the anticipation of the immediate future on the response to the same incoming stimulus. [Husserl, *On the Phenomenology of the Consciousness of Internal Time*](https://link.springer.com/book/9780792308911) · [James, *The Principles of Psychology*, chapter XV](https://psychclassics.yorku.ca/James/Principles/prin15.htm)

## Hypotheses

- **Frozen inference.** At fixed weights, tokens, token positions, inference settings, and cache state, waiting yields a probe activation within the repeatability margin. Standard transformer inference advances when a forward pass is invoked. Its key–value cache stores computed states, and its cache positions track token positions. [Vaswani et al., *Attention Is All You Need*](https://arxiv.org/abs/1706.03762) · [Hugging Face cache documentation](https://huggingface.co/docs/transformers/v4.57.0/cache_explanation)
- **Autonomous dynamics.** In a stateful agent, internal updates during a silent interval may change the probe activation. If the same initial state and ordered update sequence reproduce the effect when replayed at a different speed, the updates are sufficient to explain the observed drift under that intervention.
- **Elapsed-time sensitivity.** A difference between real waits at fixed update count and schedule would justify looking for a time-dependent pathway in the system or host. A continuous-time state transition, timer, or time-dependent memory expiry could produce that pattern. The study can identify a mechanism when intervening on that mechanism changes the effect.
- **Functional temporal integration.** A candidate temporal effect has stronger construct validity if it changes both the recoverable trace of a recent event and the prediction of an imminent event in the direction specified by the task.

## Stimulus and trial

Use short event sequences for the main assay. A sequence can establish a regularity such as `A B A B A`, then terminate in a silent interval. Start with a fixed separator token as the probe. Choose its token ID once from the pinned tokenizer, then append it directly so tokenization across a text boundary cannot alter the prefix. Within each matched trial, keep prefix IDs, probe ID, positions, and masks identical across all conditions. Supply no timestamps, padding tokens, generated text, or sensor packets during the wait. For a future multimodal replication, distinguish an absent sensor packet from a newly delivered silent or black frame.

1. Process the prefix. Synchronize outstanding work, save the resumable state, and create identical trial branches. Record the host monotonic time at the start of each assigned wait.
2. Run the assigned idle condition. The scheduler records timestamps and update counts outside the model’s observation path. Give model code an explicit clock only in the clock control. Audit timers, callbacks, memory expiry, retrieval, queues, session housekeeping, and unexpected model invocations.
3. Snapshot persistent state immediately before the probe with an identical read-only procedure in every arm. Deliver the probe and capture its post-block residual vector before any decoding or sampling. Record the actual elapsed duration to probe entry.
4. Restore the same starting state for each branch. Fingerprint what the probe can read, including tensors, recurrent memory, and random-number generator state. Abort and record a failed branch if the starting state, allowed inputs, schedule, or instrumentation differs from the assigned condition.

## Phase A · Frozen transformer baseline

Start with GPT-2 small (`openai-community/gpt2`) on CPU in FP32. Pin the model and tokenizer revisions, software versions, thread count, and attention implementation. Use evaluation and inference modes with deterministic operations enabled. Keep the model loaded in one offline process. Capture the raw block-end residual after the sixth transformer block at the probe token using a forward hook before final layer normalization. Record other block activations and logits as secondary measurements. [GPT-2 model card](https://huggingface.co/openai-community/gpt2) · [GPT-2 block implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) · [PyTorch determinism documentation](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html)

- **Resident cache.** Prefill the prefix, keep each cloned key–value cache in memory, and run no model calls during a real idle interval. Verify unchanged cache content and position before inserting the probe.
- **Recompute control.** Retain only the input IDs across the same waits, then run a fresh forward pass on the prefix and probe. Compare waits within this path, since cached and recomputed inference can use different kernels or operation shapes.
- **Intervals.** Check back-to-back repeats first. Use real 10-second and 60-second waits for the primary contrast. Add a real 10-hour interval only after the short-run protocol passes calibration and confirms stable state retention. Treat an accelerated or virtual-clock run as a mechanism control.

## Phase B · Active agent without new external input

Use an offline event-prediction agent with an explicit, checkpointable recurrent memory and a transformer readout. A concrete initial build would add a small recurrent memory to the Phase A backbone and project that memory into a reserved embedding slot after the cached prefix and immediately before the probe. Compute the slot after each wait and keep its position fixed across arms. Train the memory and readout interface on a disjoint corpus of event sequences, then freeze the learned parameters. On each low-rate idle tick, a memory-conditioned query attends to cached prefix representations and drives the same recurrent update with a fixed null input. No task event arrives during those ticks. Log the update count, state writes, and any reads beyond the cached prefix and recurrent memory. The integrated agent is the system under test in this phase.

| Condition | Intervention | What the comparison identifies |
| --- | --- | --- |
| Suspended | Zero updates at both 10 and 60 seconds. Preserve the same start state. | Sensitivity to elapsed latency while model-level state is held fixed. |
| Matched active | Eight clock-blind updates, one per second from eight seconds to one second before the probe. | Sensitivity to extra dormant time before an identical low-rate update schedule. |
| Compute dose | Hold latency at 10 seconds. Assign zero, eight, or 32 updates within the same final eight-second window. | Drift associated with update dose. Compare with the write-blocked control to account for host load. |
| Updates across the gap | Spread eight clock-blind updates across each interval. Fix the last tick one second before the probe. Replay the same ordered updates on the matched active schedule at the same total wait. | Combined timing and cadence sensitivity. The replay checks whether the causal update sequence is sufficient. |
| Write-blocked | Execute the relevant schedule and update dose on scratch memory while preserving the initial committed memory for the readout. | Whether a shift depends on the evolving memory or follows from running the loop. |
| Clock positive control | At fixed wait and update count, add a fixed nonzero embedding-direction offset to the memory slot, scaled by an injected clock value. Calibrate the direction and scale in the pilot. | Whether the assay recovers a known timing pathway and follows the injected value. |

Calibrate scheduled ticks against measured runtime before fixing the compute dose. Use the same tick count and operations in each replay and run write-blocked copies on both schedules to check for host effects. Compare final memory fingerprints as well as the residuals. If the agent requires a live clock, scheduler callbacks beyond the scheduled ticks, or sensor heartbeats, expose and log those channels as part of the architecture. Treat a continuously running condition with an update count that grows with elapsed time as a joint intervention. For localizing an effect, replay from the same snapshot and swap the final recurrent memory between matched branches before presenting the probe. A probe response that follows the transplanted memory supports mediation by that state.

## Measurement and analysis

**Primary endpoint.** For each prefix, record two independently scheduled replicate pairs for each contrast. Let `A1`, `A2` be the two control residual vectors and `B1`, `B2` their matched intervention vectors. Use the FP64 dot product of `(B1 - A1)` and `(B2 - A2)`, divided by the squared norm of a separate immediate reference response for that prefix. Fix the reference before assigning branches and apply a `1e-12` floor to its norm. With independent repeat errors, this estimates squared mean displacement without treating greater variance in one arm as a reproducible shift. Sampling estimates can be negative. Also report the normalized distances between same-condition repeats for each arm.

**Prespecified contrasts.** Compare 60 seconds with 10 seconds in the frozen resident-cache arm, then in the matched-active arm. At a fixed 10 seconds, compare eight committed updates with zero updates as the main mechanistic contrast. Compare the eight-update and eight-update write-blocked arms to check whether committed memory changes explain that effect. Report the write-blocked comparison, replay, and the 32-update arm as mechanism diagnostics.

**Sampling and assignment.** Pilot with 16 prefixes and four independent sessions. Use 48 disjoint, fixed prefixes for the first confirmatory run with 16 independent sessions per phase. Each session uses a fresh process and measures the full panel. Randomly assign conditions to matched branches and counterbalance branch launch order. Schedule short repeats both before and after longer waits across sessions to expose warm-up and host drift. Average within session, then estimate uncertainty across sessions. Prefixes sharing the same timing run and activations from different layers are repeated measurements. Freeze the stimulus file, schedule, endpoint, and analysis commit after the pilot and before collecting the confirmatory runs. Check feasibility and expected precision using the pilot, and record any sample-size change before confirmation.

**Decision rule.** Resample whole sessions in 10,000 bootstrap draws with a fixed analysis seed. Report two-sided 98.33% intervals for the three prespecified contrasts, using a Bonferroni adjustment for the family. Start with a provisional engineering margin of `1e-4` normalized distance, or `1e-8` on the squared endpoint. In the pilot, verify same-condition repeat distances stay below the distance margin and recover a known `1e-3` normalized displacement injected consistently in one arm. Justify and lock the margin before confirmation. A confirmatory interval wholly above the positive squared margin supports reproducible drift. An interval wholly within the symmetric squared-margin band supports equivalence at the specified resolution. Classify other intervals as inconclusive. Report all condition means, repeat distances, realized waits, and exclusions. A failed control triggers a diagnostic rerun with the original record retained.

**Secondary measurements.** Report cosine distance, norm change, and the differences between pairwise prefix-distance matrices across conditions. Per-vector displacement can coexist with stable geometry across prefixes. Plot effect sizes and intervals by block and latency as exploratory results. Compare state fingerprints and output logits to determine whether a residual shift propagates to the readout.

**Retention and protention follow-up.** Cross a repeat or alternate rule with a final event of `A` or `B`, using held-out sequences. For retention, use the neutral separator and measure recovery of the last event with a fixed classifier trained on disjoint prefixes. Balance the predicted next event within each last-event class. For protention, read next-event probabilities from a clone of the pre-probe state and measure the surprisal gap between expected and unexpected occurrences of the same event. For example, after `A`, a cue of `B` is expected under alternation and unexpected under repetition. Counterbalance rules and cue identities and compare the surprisal gap across waits. A fading-state hypothesis predicts weaker last-event recovery and a smaller expectancy gap at longer waits. Verify short-wait task competence before interpreting either effect.

## Interpretation

| Result | Supported conclusion |
| --- | --- |
| Frozen and matched-active waits are equivalent within the margin | The tested waits, schedules, and readout show no detectable latency effect of the specified size. |
| Drift follows update dose, disappears when writes are blocked, and is reproduced by replay | Autonomous computation during the gap changes how the system responds. |
| Drift follows a spoofed clock value at fixed real latency | The injected clock pathway is sufficient to produce temporal sensitivity. |
| Drift survives matched updates, replay, and the host controls | A residual latency association merits independent replication and further mechanism tracing. |
| Drift also changes recent-event recovery and expectancy effects as predicted | The temporal dynamics have measurable consequences for how the system integrates recent and imminent events. |

Even a simple clock-driven or leaky memory can produce delay-dependent geometry. The experiment establishes which temporal variables affect a defined readout and can help locate the responsible mechanism. Whether that mechanism is accompanied by a lived experience of duration remains underdetermined. [Butlin et al., *Consciousness in Artificial Intelligence*](https://arxiv.org/abs/2308.08708)

## Reproducibility and execution order

Record per-session hardware, OS, package and model revisions, seeds, thread settings, and config hashes. For each branch, save the stimulus ID, arm, target and actual wait, tick times and counts, clock inputs, start and pre-probe state fingerprints, probe activations, logits, and failure flags. Predefine exclusion criteria for drift in the delivered schedule, process restarts, cache eviction, and unexpected inputs. Save the reason for every excluded matched set and report attrition by arm.

First demonstrate deterministic back-to-back capture and recovery of the injected signal. Then collect the short frozen and active comparisons. Proceed to the real 10-hour extension once the host can preserve state and log interruptions unattended. Pair it with short frozen and matched-update controls on the same host, and repeat across independent sessions before drawing an inference about that interval. Keep raw pilot and confirmatory records alongside the exact protocol revision used to generate them.
