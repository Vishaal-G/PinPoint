"""Entry point for the Databricks App. Streamlit reruns this file on every interaction, and a plain
`import pinpoint.app` would only execute the page once (modules are cached), so run it fresh each time."""
import runpy

runpy.run_module("pinpoint.app", run_name="__main__")
