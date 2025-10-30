from __future__ import annotations
import argparse
import json
from .embedlib import op_embed, topk_search, analyze_index


def main():
    parser = argparse.ArgumentParser(prog="embedlab")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("embed")
    p1.add_argument("--images-dir", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility"
    )
    p1.add_argument(
        "--batch-size", type=int, default=1, help="batch size for encoding images"
    )

    p2 = sub.add_parser("search")
    p2.add_argument("--index", required=True)
    p2.add_argument("--query-dir", required=True)
    p2.add_argument("--k", type=int, default=5)
    p2.add_argument("--json", action="store_true")
    p2.add_argument(
        "--batch-size", type=int, default=1, help="batch size for encoding query images"
    )
    p2.add_argument(
        "--seed", type=int, default=42, help="random seed used when encoding queries"
    )

    p3 = sub.add_parser("analyze")
    p3.add_argument("--index", required=True)
    p3.add_argument("--dup-threshold", type=float, required=True)
    p3.add_argument("--anomaly-top", type=int, required=True)
    p3.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.cmd == "embed":
        op_embed(args.images_dir, args.out, seed=args.seed, batch_size=args.batch_size)
    elif args.cmd == "search":
        res = topk_search(
            args.index,
            args.query_dir,
            args.k,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        if args.json:
            for r in res:
                print(json.dumps(r))
        else:
            for r in res:
                print("Query:", r["query"])
                for it in r["results"]:
                    print(f"  {it['path']} score={it['score']:.4f}")
    elif args.cmd == "analyze":
        out = analyze_index(args.index, args.dup_threshold, args.anomaly_top)
        if args.json:
            print(json.dumps(out))
        else:
            print(out)
