"""add closed-loop learning tables — edge_log, drift_snapshots, agent_evolution, bankroll_ledger

Revision ID: a1b2c3d4e5f6
Revises: 5c5d124c4ef3
Create Date: 2026-04-02 21:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '5c5d124c4ef3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create closed-loop learning tables."""

    # Edge discovery log
    op.create_table(
        'edge_log',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('agent_name', sa.String(length=100), nullable=False),
        sa.Column('sport_slug', sa.String(length=50), nullable=False),
        sa.Column('league', sa.String(length=200), nullable=False),
        sa.Column('market', sa.String(length=50), nullable=False),
        sa.Column('edge_type', sa.String(length=50), nullable=False),
        sa.Column('edge_value', sa.Float(), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('confidence_interval_lo', sa.Float(), nullable=True),
        sa.Column('confidence_interval_hi', sa.Float(), nullable=True),
        sa.Column('is_significant', sa.Boolean(), nullable=True, default=False),
        sa.Column('discovered_date', sa.Date(), nullable=False),
        sa.Column('expired_date', sa.Date(), nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_edge_log_date', 'edge_log', ['discovered_date'])
    op.create_index('ix_edge_log_agent', 'edge_log', ['agent_name'])

    # Model drift snapshots
    op.create_table(
        'drift_snapshots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('sport_slug', sa.String(length=50), nullable=False),
        sa.Column('snapshot_date', sa.Date(), nullable=False),
        sa.Column('window_days', sa.Integer(), nullable=False, default=30),
        sa.Column('calibration_error', sa.Float(), nullable=False),
        sa.Column('brier_score', sa.Float(), nullable=False),
        sa.Column('log_loss', sa.Float(), nullable=True),
        sa.Column('n_predictions', sa.Integer(), nullable=False),
        sa.Column('accuracy', sa.Float(), nullable=False),
        sa.Column('roi_pct', sa.Float(), nullable=True),
        sa.Column('clv_mean', sa.Float(), nullable=True),
        sa.Column('drift_detected', sa.Boolean(), nullable=True, default=False),
        sa.Column('drift_severity', sa.String(length=20), nullable=True),
        sa.Column('retrain_triggered', sa.Boolean(), nullable=True, default=False),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_drift_date', 'drift_snapshots', ['snapshot_date'])

    # Agent evolution audit trail
    op.create_table(
        'agent_evolution',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('agent_name', sa.String(length=100), nullable=False),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('parent_agent', sa.String(length=100), nullable=True),
        sa.Column('event_date', sa.Date(), nullable=False),
        sa.Column('config_before', sa.JSON(), nullable=True),
        sa.Column('config_after', sa.JSON(), nullable=True),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('performance_at_event', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_agent_evo_date', 'agent_evolution', ['event_date'])

    # Bankroll audit ledger (immutable)
    op.create_table(
        'bankroll_ledger',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('event_date', sa.Date(), nullable=False),
        sa.Column('event_type', sa.String(length=30), nullable=False),
        sa.Column('pick_id', sa.Integer(), sa.ForeignKey('daily_picks.id'), nullable=True),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('bankroll_before', sa.Float(), nullable=False),
        sa.Column('bankroll_after', sa.Float(), nullable=False),
        sa.Column('drawdown_pct', sa.Float(), nullable=False, default=0.0),
        sa.Column('circuit_breaker_level', sa.Integer(), nullable=False, default=0),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_bankroll_date', 'bankroll_ledger', ['event_date'])


def downgrade() -> None:
    """Drop closed-loop learning tables."""
    op.drop_table('bankroll_ledger')
    op.drop_table('agent_evolution')
    op.drop_table('drift_snapshots')
    op.drop_table('edge_log')
