import asyncio
import argparse

from src.flow import AppFlow


def get_args():
    parser = argparse.ArgumentParser(description="Run the app flow.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The query to process.",
    )
    return parser.parse_args()


async def main(args):
    app_flow = AppFlow(timeout=1000, verbose=False)
    result = await app_flow.run(query=args.query)
    return result


if __name__ == "__main__":
    args = get_args()

    asyncio.run(main(args))
