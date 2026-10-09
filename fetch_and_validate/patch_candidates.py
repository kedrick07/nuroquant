import pandas as pd

df = pd.read_csv("data/universe/candidates.csv")

if "aliases" not in df.columns:
    df["aliases"] = ""
df["aliases"] = df["aliases"].fillna("")

fixes = {
    "LLY":   {"company_name": "Eli Lilly and Company", "add_aliases": ["Eli Lilly", "Lilly"]},
    "META":  {"company_name": "Meta Platforms Inc",     "add_aliases": ["Facebook", "Meta"]},
    "MA":    {"company_name": "Mastercard Inc",          "add_aliases": ["Mastercard"]},
    "PG":    {"company_name": "Procter & Gamble Co",     "add_aliases": ["Procter & Gamble", "P&G"]},
    "HD":    {"company_name": "Home Depot Inc",          "add_aliases": ["Home Depot"]},
    "UNH":   {"company_name": "UnitedHealth Group Inc",  "add_aliases": ["UnitedHealth", "United Health"]},
    "BRK.B": {"company_name": "Berkshire Hathaway Inc",  "add_aliases": ["Berkshire Hathaway", "Berkshire"]},
    "AVGO":  {"company_name": "Broadcom Inc",            "add_aliases": ["Broadcom"]},
}

for ticker, fix in fixes.items():
    mask = df["ticker"] == ticker
    if not mask.any():
        print(f"WARNING: {ticker} not found in candidates.csv")
        continue
    old_name = df.loc[mask, "company_name"].values[0]
    df.loc[mask, "company_name"] = fix["company_name"]
    existing = [a.strip() for a in df.loc[mask, "aliases"].values[0].split("|") if a.strip()]
    combined = existing + [a for a in fix["add_aliases"] if a not in existing]
    df.loc[mask, "aliases"] = "|".join(combined)
    print(f"{ticker}: '{old_name}' -> '{fix['company_name']}' | aliases: {df.loc[mask, 'aliases'].values[0]}")

df.to_csv("data/universe/candidates.csv", index=False)
print("\nSaved.")