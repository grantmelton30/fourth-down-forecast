"""Record a real book quote; no wagers are placed by this command."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"shared"))
from execution_quotes import append_quote


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("league","game-id","kickoff","book","source","observed-at"):
        p.add_argument("--"+key,required=True)
    for key in ("line","over-price","under-price"):
        p.add_argument("--"+key,required=True,type=float)
    p.add_argument("--ledger",type=Path,default=ROOT/"data/execution_quotes.jsonl")
    args=vars(p.parse_args())
    row=append_quote(args.pop("ledger"),args)
    print(json.dumps({"quote_id":row["quote_id"],"status":"recorded; execution unverified"}))


if __name__=="__main__":
    main()
