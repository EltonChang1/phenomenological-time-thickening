# Phenomenological Time-Thickening Test

How do AI feel time? Certainly not continuous but discrete, but how?
A Husserl-inspired experiment proposal for testing how an AI system’s internal state changes during a period without new external input. The experiment pairs identical one-token probes across real idle intervals and measures the resulting transformer residual-stream activations.

[Read the experiment proposal](PROPOSAL.md)

The study starts with a frozen GPT-2 baseline, then tests a stateful agent whose elapsed wait, background updates, and access to clocks can be varied independently. A follow-up asks whether any drift changes how the agent carries forward recent events and anticipates the next event.

**Status** · Proposal. Implementation and data collection are planned. The experiment measures functional temporal dynamics. Subjective experience remains an open question under every proposed outcome.
