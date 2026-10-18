"""Alembic environment: uses DATABASE_URL and the app's SQLAlchemy metadata."""

import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv

from src.services import models  # noqa: F401  # register tables on Base.metadata
from src.services.db import Base, make_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
load_dotenv()
database_url = os.getenv("DATABASE_URL", "sqlite:///./finance_agent.db")


def run_migrations_offline() -> None:
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=database_url.startswith("sqlite"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(database_url)
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
