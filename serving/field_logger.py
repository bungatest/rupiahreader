"""
Field Test Logger for Rupiah Serving.
Records predictions, ground-truth labels provided during field testing,
and exports evaluation metrics and CSV logs.
"""

import sqlite3
import csv
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from .config import FIELD_LOG_PATH, CLASSES, CLASS_DISPLAY_NAMES


class FieldLogger:
    def __init__(self, db_path: Path = FIELD_LOG_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS field_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    ground_truth TEXT,
                    confidence REAL NOT NULL,
                    is_correct INTEGER,
                    latency_ms REAL NOT NULL,
                    agreement_consensus TEXT,
                    notes TEXT
                )
            """)
            conn.commit()

    def log_prediction(
        self,
        model_used: str,
        predicted_class: str,
        confidence: float,
        latency_ms: float,
        ground_truth: Optional[str] = None,
        agreement: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        now = datetime.utcnow().isoformat()
        is_correct = None
        if ground_truth:
            is_correct = 1 if predicted_class == ground_truth else 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO field_predictions (
                    timestamp, model_used, predicted_class, ground_truth,
                    confidence, is_correct, latency_ms, agreement_consensus, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    model_used,
                    predicted_class,
                    ground_truth,
                    confidence,
                    is_correct,
                    latency_ms,
                    str(agreement) if agreement is not None else None,
                    notes,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def update_ground_truth(self, log_id: int, ground_truth: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT predicted_class FROM field_predictions WHERE id = ?",
                (log_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False
            pred = row["predicted_class"]
            is_correct = 1 if pred == ground_truth else 0
            cursor.execute(
                """
                UPDATE field_predictions
                SET ground_truth = ?, is_correct = ?
                WHERE id = ?
                """,
                (ground_truth, is_correct, log_id),
            )
            conn.commit()
            return True

    def get_summary_report(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Overall
            cursor.execute("SELECT COUNT(*) FROM field_predictions")
            total_logs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM field_predictions WHERE ground_truth IS NOT NULL")
            verified_logs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM field_predictions WHERE is_correct = 1")
            correct_logs = cursor.fetchone()[0]

            acc = (correct_logs / verified_logs) if verified_logs > 0 else 0.0

            # Per model stats
            cursor.execute("""
                SELECT 
                    model_used,
                    COUNT(*) as total,
                    SUM(CASE WHEN ground_truth IS NOT NULL THEN 1 ELSE 0 END) as verified,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(latency_ms) as avg_latency,
                    AVG(confidence) as avg_conf
                FROM field_predictions
                GROUP BY model_used
            """)
            model_stats = {}
            for row in cursor.fetchall():
                m_total = row["total"]
                m_ver = row["verified"] or 0
                m_corr = row["correct"] or 0
                m_acc = (m_corr / m_ver) if m_ver > 0 else None
                model_stats[row["model_used"]] = {
                    "total": m_total,
                    "verified": m_ver,
                    "correct": m_corr,
                    "accuracy": round(m_acc, 4) if m_acc is not None else None,
                    "avg_latency_ms": round(row["avg_latency"] or 0, 1),
                    "avg_confidence": round(row["avg_conf"] or 0, 4),
                }

            # Per class stats
            cursor.execute("""
                SELECT 
                    ground_truth,
                    COUNT(*) as verified,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM field_predictions
                WHERE ground_truth IS NOT NULL
                GROUP BY ground_truth
            """)
            class_stats = {}
            for row in cursor.fetchall():
                c_ver = row["verified"]
                c_corr = row["correct"]
                gt = row["ground_truth"]
                class_stats[gt] = {
                    "display": CLASS_DISPLAY_NAMES.get(gt, gt),
                    "verified": c_ver,
                    "correct": c_corr,
                    "recall": round(c_corr / c_ver, 4) if c_ver > 0 else 0.0,
                }

            # Recent 10 logs
            cursor.execute("""
                SELECT id, timestamp, model_used, predicted_class, ground_truth, confidence, is_correct, latency_ms
                FROM field_predictions
                ORDER BY id DESC
                LIMIT 10
            """)
            recent = [dict(row) for row in cursor.fetchall()]

            return {
                "total_inferences": total_logs,
                "verified_tests": verified_logs,
                "correct_tests": correct_logs,
                "overall_accuracy": round(acc, 4) if verified_logs > 0 else None,
                "per_model": model_stats,
                "per_class": class_stats,
                "recent_logs": recent,
            }

    def export_csv(self) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM field_predictions ORDER BY id ASC")
            rows = cursor.fetchall()

            output = io.StringIO()
            writer = csv.writer(output)
            if rows:
                writer.writerow(rows[0].keys())
                for r in rows:
                    writer.writerow(list(r))
            else:
                writer.writerow([
                    "id", "timestamp", "model_used", "predicted_class",
                    "ground_truth", "confidence", "is_correct", "latency_ms",
                    "agreement_consensus", "notes"
                ])
            return output.getvalue()

