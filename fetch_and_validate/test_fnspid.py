import pandas as pd

df = pd.read_csv("data/news/fnspid/processed/fnspid_monthly_counts.csv")  # adjust path
tickers = ["BRK.B", "UNH", "MA", "PG", "HD", "AVGO"]

pivot = df[df["ticker"].isin(tickers)].pivot_table(
    index="year_month", columns="ticker", values="article_count", fill_value=0
)

zero_mask = pivot == 0
overlap = zero_mask.sum(axis=1).sort_values(ascending=False)
print("Months where multiple tickers have zero articles (higher = more overlap):")
print(overlap[overlap >= 3])

pivot.to_csv("data/news/fnspid/processed/fnspid_zero_month_check.csv")
zero_mask.astype(int).to_csv("data/news/fnspid/processed/fnspid_zero_mask.csv")