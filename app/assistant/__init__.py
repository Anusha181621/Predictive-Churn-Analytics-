"""The "Ask the Data" assistant: a Claude agent that answers questions from the artefacts.

The package is split so that everything worth testing can be tested without a network call:

* :mod:`app.assistant.tools` holds the six tool functions and their declarations. They are ordinary
  pandas over the same master frame the pages render, with no provider types anywhere in them, and
  they are where every number the assistant reports comes from.
* :mod:`app.assistant.agent` holds the model wiring -- client, system prompt, tool loop -- and is
  the only module in the project that imports ``openai``. Keeping the provider confined to this one
  file is what made switching providers a change to it alone.

Nothing under ``src/`` imports either one. The pipeline that computes features, churn probability,
explanations and retention economics stays entirely offline; the assistant reads what that pipeline
already wrote and cannot change a single figure.
"""
