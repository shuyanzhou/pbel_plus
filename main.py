from config import argps
from models import charagram, lstm, charcnn

args = argps()

if args.model == "charagram":
    charagram.main(args)
elif args.model in ["lstm", "avg_lstm"]:
    lstm.main(args)
elif args.model == "charcnn":
    charcnn.main(args)
