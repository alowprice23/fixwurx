#!/usr/bin/env python3
"""
Setup script for the Neural Matrix database.
"""

import sqlite3
import json
import time
import os
from pathlib import Path

# Paths
db_path = '.triangulum/neural_matrix/patterns/patterns.db'
patterns_path = '.triangulum/neural_matrix/patterns/starter_patterns.json'

# Create database
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Create tables
cur.execute('''
CREATE TABLE IF NOT EXISTS neural_patterns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id   TEXT    NOT NULL,
    bug_type     TEXT    NOT NULL,
    tags         TEXT    NOT NULL,
    features     TEXT    NOT NULL,
    success_rate REAL    NOT NULL DEFAULT 0.0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    created_at   REAL    NOT NULL,
    updated_at   REAL    NOT NULL
)
''')

cur.execute('''
CREATE TABLE IF NOT EXISTS neural_weights (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    weight_key   TEXT    NOT NULL,
    weight_value REAL    NOT NULL,
    category     TEXT    NOT NULL,
    description  TEXT,
    created_at   REAL    NOT NULL,
    updated_at   REAL    NOT NULL
)
''')

# Get current time
now = time.time()

# Load patterns from file
with open(patterns_path, 'r') as f:
    patterns = json.load(f)

# Insert patterns
for pattern in patterns:
    cur.execute(
        '''
        INSERT INTO neural_patterns
        (pattern_id, bug_type, tags, features, success_rate, sample_count, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            pattern['pattern_id'],
            pattern['bug_type'],
            json.dumps(pattern['tags']),
            json.dumps(pattern['features']),
            pattern['success_rate'],
            pattern['sample_count'],
            now,
            now
        )
    )

# Insert weights
weight_data = [
    ('observer', 1.0, 'agent', 'Observer agent weight', now, now),
    ('analyst', 1.2, 'agent', 'Analyst agent weight', now, now),
    ('verifier', 0.9, 'agent', 'Verifier agent weight', now, now),
    ('planner', 1.5, 'agent', 'Planner agent weight', now, now),
    ('pattern_match', 1.5, 'feature', 'Pattern matching weight', now, now),
    ('entropy', 0.8, 'feature', 'Entropy calculation weight', now, now),
    ('fallback', 0.5, 'feature', 'Fallback strategy weight', now, now),
    ('similarity', 1.2, 'feature', 'Similarity calculation weight', now, now)
]

for weight in weight_data:
    cur.execute(
        '''
        INSERT INTO neural_weights
        (weight_key, weight_value, category, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''',
        weight
    )

# Commit changes and close connection
conn.commit()
conn.close()

print('Database created successfully!')
print(f'Database path: {os.path.abspath(db_path)}')
print(f'Patterns inserted: {len(patterns)}')
print(f'Weights inserted: {len(weight_data)}')
