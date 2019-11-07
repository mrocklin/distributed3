import asyncio
import click
import json
import yaml

from distributed.deploy.spec import run_workers


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument("scheduler", type=str, required=False)
@click.option("--spec", type=str, default="", help="")
@click.option("--spec-file", type=str, default=None, help="")
@click.version_option()
def main(scheduler: str, spec: str, spec_file: str):
    _spec = {}
    if spec_file:
        with open(spec_file) as f:
            _spec.update(yaml.safe_load(f))

    if spec:
        _spec.update(json.loads(spec))

    if "cls" in _spec:  # single worker spec
        _spec = {_spec["opts"].get("name", 0): _spec}

    async def run():
        workers = await run_workers(scheduler, _spec)
        try:
            await asyncio.gather(*[w.finished() for w in workers.values()])
        except KeyboardInterrupt:
            await asyncio.gather(*[w.close() for w in workers.values()])

    asyncio.get_event_loop().run_until_complete(run())


if __name__ == "__main__":
    main()
