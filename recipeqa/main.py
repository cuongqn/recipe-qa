import argparse
import logging

import recipe

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


def main(args):
    question = args.question
    agent = recipe.Agent()
    answer = agent(query=question, fetch_k=args.fetch_k, top_k=args.top_k)
    print(f"Question: {question}\n")
    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question")
    parser.add_argument("--fetch_k", default=20, type=int)
    parser.add_argument("--top_k", default=5, type=int)
    args = parser.parse_args()
    main(args)
