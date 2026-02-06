from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg://ocean_user:ocean_secure_password@127.0.0.1:5432/ocean_db",
    pool_pre_ping=True,
)

with engine.connect() as conn:
    print(conn.execute(text("SELECT COUNT(*) FROM ocean_metrics")).scalar())
