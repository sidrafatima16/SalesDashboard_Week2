
import numpy as np
import pandas as pd

RAW_FILE = "raw_sales_data.csv"
CLEAN_FILE = "cleaned_sales_data.csv"
DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%B %d, %Y"]
TEXT_COLS = ["Customer Name", "Product", "Category", "Region"]

pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


# ---------------------------------------------------------------- load / inspect
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def inspect(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print("Shape:", df.shape)
    df.info()
    print("\nNumeric summary:\n", df.describe())
    print("\nMissing values:\n", df.isna().sum().to_string())
    print("Duplicate rows:", df.duplicated().sum())


# ---------------------------------------------------------------- cleaning steps
def standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Trim spaces and fix casing: ' europe ' / 'EUROPE' -> 'Europe'."""
    for col in TEXT_COLS:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.title()
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Try each known format in turn; rows that match none become NaT and are dropped."""
    s = df["Order Date"].astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        todo = parsed.isna()
        parsed[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    bad = parsed.isna().sum()
    if bad:
        print(f"Dropping {bad} rows with unreadable dates")
    df["Order Date"] = parsed
    return df.dropna(subset=["Order Date"]).reset_index(drop=True)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute instead of deleting, so real orders are not lost.
    Sales  = Quantity x median unit price of that product.
    Profit = Sales x median profit margin of that category."""
    df["Sales Imputed"] = df["Sales"].isna()
    df["Profit Imputed"] = df["Profit"].isna()

    unit_price = (df["Sales"] / df["Quantity"]).groupby(df["Product"]).median()
    m = df["Sales"].isna()
    df.loc[m, "Sales"] = (df.loc[m, "Quantity"] * df.loc[m, "Product"].map(unit_price)).round(2)

    margin = (df["Profit"] / df["Sales"]).groupby(df["Category"]).median()
    m = df["Profit"].isna()
    df.loc[m, "Profit"] = (df.loc[m, "Sales"] * df.loc[m, "Category"].map(margin)).round(2)

    print(f"Imputed {int(df['Sales Imputed'].sum())} Sales and {int(df['Profit Imputed'].sum())} Profit values")
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Sales"] >= 0)].reset_index(drop=True)
    print(f"Removed {before - len(df)} invalid rows (Quantity <= 0 or negative Sales)")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)
    df["Month Num"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.strftime("%b")
    df["Quarter"] = "Q" + df["Order Date"].dt.quarter.astype(str)
    df["Profit Margin %"] = (df["Profit"] / df["Sales"] * 100).round(2)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_text(df)
    df = remove_duplicates(df)
    df = parse_dates(df)
    df = handle_missing(df)
    df = validate(df)
    return add_features(df)


# ---------------------------------------------------------------- KPIs & EDA
def calculate_kpis(df: pd.DataFrame) -> dict:
    total_sales = df["Sales"].sum()
    orders = df["Order ID"].nunique()
    return {
        "Total Sales": total_sales,
        "Total Profit": df["Profit"].sum(),
        "Total Orders": orders,
        "Total Quantity": int(df["Quantity"].sum()),
        "Average Order Value": total_sales / orders,
        "Overall Profit Margin %": df["Profit"].sum() / total_sales * 100,
    }


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    t = df.groupby(by).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique"),
                           Quantity=("Quantity", "sum"))
    t["Margin %"] = t["Profit"] / t["Sales"] * 100
    t["Sales Share %"] = t["Sales"] / t["Sales"].sum() * 100
    return t.sort_values("Sales", ascending=False)


def run_eda(df: pd.DataFrame) -> None:
    print("\n" + "=" * 20 + " KEY METRICS " + "=" * 20)
    for k, v in calculate_kpis(df).items():
        print(f"{k:<26}{v:>14,.2f}")

    print("\nBy Category:\n", summarize(df, "Category"))
    print("\nBy Region:\n", summarize(df, "Region"))
    print("\nTop 10 Products by Sales:\n", summarize(df, "Product").head(10))
    print("\nBy Product (ranked by margin):\n", summarize(df, "Product").sort_values("Margin %", ascending=False)[["Sales", "Profit", "Margin %"]])
    print("\nBy Quarter:\n", summarize(df, "Quarter").sort_index())
    monthly = summarize(df, "Month").sort_index()
    monthly["MoM Sales Growth %"] = monthly["Sales"].pct_change() * 100
    print("\nMonthly trend:\n", monthly[["Sales", "Profit", "Margin %", "MoM Sales Growth %"]])


def main() -> None:
    raw = load_data(RAW_FILE)
    inspect(raw, "RAW DATA")
    clean = clean_data(raw.copy())
    inspect(clean, "CLEANED DATA")
    run_eda(clean)
    clean["Order Date"] = clean["Order Date"].dt.strftime("%Y-%m-%d")
    clean.to_csv(CLEAN_FILE, index=False)
    print(f"\nSaved {len(clean):,} rows to {CLEAN_FILE}")


if __name__ == "__main__":
    main()
