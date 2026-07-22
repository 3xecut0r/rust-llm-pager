# Deliberately no eager re-exports here: .app (and transitively .__main__)
# import fastapi/uvicorn, which are an optional "serve" extra, not a core
# pager_hf dependency. Import what you need directly:
#   from pager_hf.serving.scheduler import ContinuousBatchingScheduler
#   from pager_hf.serving.app import create_app
# so `import pager_hf.serving.scheduler` alone (e.g. from tests) never
# requires fastapi/uvicorn to be installed.
