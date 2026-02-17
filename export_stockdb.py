import os
import pandas as pd
from sqlalchemy import create_engine, text

# SQLAlchemy connection string: mysql+pymysql://user:password@host:port/database
DB_URL = "mysql+pymysql://root:comeOn#200@10.0.0.205:3306/stockdb"
engine = create_engine(DB_URL)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("Connecting to MariaDB stockdb...")
    with engine.connect() as conn:
        # 1. List all tables
        result = conn.execute(text("SHOW TABLES"))
        tables = [t[0] for t in result.fetchall()]
        print(f"\n{'='*60}")
        print(f"Found {len(tables)} tables: {tables}")
        print(f"{'='*60}")

        # 2. For each table, describe schema and stats
        for tname in tables:
            print(f"\n{'='*60}")
            print(f"TABLE: {tname}")
            print(f"{'='*60}")

            # Schema
            result = conn.execute(text(f"DESCRIBE `{tname}`"))
            cols = result.fetchall()
            print(f"\n  Columns:")
            col_names = []
            for c in cols:
                col_names.append(c[0])
                nullable = "NULL" if c[2] == "YES" else "NOT NULL"
                key = f"[{c[3]}]" if c[3] else ""
                print(f"    {c[0]:35s} {str(c[1]):25s} {nullable:10s} {key}")

            # Row count
            result = conn.execute(text(f"SELECT COUNT(*) FROM `{tname}`"))
            count = result.fetchone()[0]
            print(f"\n  Total rows: {count:,}")

            # Try to find date/datetime columns and show range
            date_cols = [c[0] for c in cols if 'date' in str(c[1]).lower() or 'time' in str(c[1]).lower()]
            for dc in date_cols:
                try:
                    result = conn.execute(text(f"SELECT MIN(`{dc}`), MAX(`{dc}`) FROM `{tname}`"))
                    mn, mx = result.fetchone()
                    print(f"  Date range ({dc}): {mn} → {mx}")
                except:
                    pass

            # Show sample rows
            result = conn.execute(text(f"SELECT * FROM `{tname}` LIMIT 5"))
            rows = result.fetchall()
            print(f"\n  Sample rows (first 5):")
            print(f"    Cols: {col_names}")
            for r in rows:
                print(f"    {r}")

            # Show indexes
            try:
                result = conn.execute(text(f"SHOW INDEX FROM `{tname}`"))
                indexes = result.fetchall()
                if indexes:
                    print(f"\n  Indexes:")
                    for idx in indexes:
                        print(f"    {idx[2]:20s} col={idx[4]} unique={not idx[1]}")
            except:
                pass

    # 3. Export all tables to CSV
    print(f"\n{'='*60}")
    print(f"EXPORTING TO CSV...")
    print(f"{'='*60}")

    with engine.connect() as conn:
        for tname in tables:
            outpath = os.path.join(OUTPUT_DIR, f"{tname}.csv")
            print(f"\n  Exporting {tname}...", end=" ", flush=True)

            # Use chunked reading for large tables
            chunk_size = 100_000
            result = conn.execute(text(f"SELECT COUNT(*) FROM `{tname}`"))
            total = result.fetchone()[0]

            if total == 0:
                print("(empty, skipping)")
                continue

            # Export in chunks
            written = 0
            first_chunk = True
            for offset in range(0, total, chunk_size):
                query = text(f"SELECT * FROM `{tname}` LIMIT {chunk_size} OFFSET {offset}")
                df = pd.read_sql(query, conn)
                df.to_csv(outpath, mode='w' if first_chunk else 'a',
                          header=first_chunk, index=False)
                first_chunk = False
                written += len(df)
                print(f"{written:,}/{total:,}", end=" ", flush=True)

            file_size = os.path.getsize(outpath) / (1024*1024)
            print(f"→ {outpath} ({file_size:.1f} MB)")

    print(f"\n{'='*60}")
    print("DONE! All tables exported to CSV.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
