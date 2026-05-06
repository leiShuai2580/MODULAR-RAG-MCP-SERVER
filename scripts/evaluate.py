#!/usr/bin/env python
"""Evaluation script for Modular RAG MCP Server. / Modular RAG MCP Server 的评估脚本。

Runs batch evaluation against a golden test set and outputs a metrics report. / 基于黄金测试集运行批量评估并输出指标报告。

Usage: / 用法：
    # Run with default settings (custom evaluator) / 使用默认设置运行（custom evaluator）
    python scripts/evaluate.py

    # Specify a custom golden test set / 指定自定义黄金测试集
    python scripts/evaluate.py --test-set path/to/golden.json

    # Use a specific collection / 使用指定集合
    python scripts/evaluate.py --collection technical_docs

    # JSON output / JSON 输出
    python scripts/evaluate.py --json

Exit codes: / 退出码：
    0 - Success / 成功
    1 - Evaluation failure / 评估失败
    2 - Configuration error / 配置错误
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console / 为 Windows 控制台设置 UTF-8 编码
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add project root to path / 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. / 解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation against a golden test set."
    )
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="Path to golden test set JSON file (default: tests/fixtures/golden_test_set.json)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name to search within.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Skip retrieval (evaluate with mock chunks for testing).",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point. / 主入口点。"""
    args = parse_args()

    try:
        from src.core.settings import load_settings
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory
        from src.observability.evaluation.eval_runner import EvalRunner

        settings = load_settings()
    except Exception as exc:
        print(f"❌ Configuration error: {exc}", file=sys.stderr)
        return 2

    # Create evaluator from config / 根据配置创建评估器
    try:
        evaluator = EvaluatorFactory.create(settings)
        evaluator_name = type(evaluator).__name__
    except Exception as exc:
        print(f"❌ Failed to create evaluator: {exc}", file=sys.stderr)
        return 2

    # Create HybridSearch (unless --no-search) / 创建 HybridSearch（除非指定 --no-search）
    hybrid_search = None
    if not args.no_search:
        try:
            from src.core.query_engine.query_processor import QueryProcessor
            from src.core.query_engine.hybrid_search import create_hybrid_search
            from src.core.query_engine.dense_retriever import create_dense_retriever
            from src.core.query_engine.sparse_retriever import create_sparse_retriever
            from src.ingestion.storage.bm25_indexer import BM25Indexer
            from src.libs.embedding.embedding_factory import EmbeddingFactory
            from src.libs.vector_store.vector_store_factory import VectorStoreFactory

            collection = args.collection or "default"

            vector_store = VectorStoreFactory.create(
                settings, collection_name=collection,
            )
            embedding_client = EmbeddingFactory.create(settings)
            dense_retriever = create_dense_retriever(
                settings=settings,
                embedding_client=embedding_client,
                vector_store=vector_store,
            )
            bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
            sparse_retriever = create_sparse_retriever(
                settings=settings,
                bm25_indexer=bm25_indexer,
                vector_store=vector_store,
            )
            sparse_retriever.default_collection = collection

            query_processor = QueryProcessor()
            hybrid_search = create_hybrid_search(
                settings=settings,
                query_processor=query_processor,
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
            )
            print(f"✅ HybridSearch initialized for collection: {collection}")
        except Exception as exc:
            print(f"⚠️  Failed to initialize search (running without retrieval): {exc}")

    # Create and run EvalRunner / 创建并运行 EvalRunner
    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
    )

    try:
        print(f"\n🔍 Running evaluation with {evaluator_name}...")
        print(f"📄 Test set: {args.test_set}")
        print(f"🔢 Top-K: {args.top_k}\n")

        report = runner.run(
            test_set_path=args.test_set,
            top_k=args.top_k,
            collection=args.collection,
        )
    except Exception as exc:
        print(f"❌ Evaluation failed: {exc}", file=sys.stderr)
        return 1

    # Output results / 输出结果
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_report(report)

    return 0


def _print_report(report) -> None:
    """Print formatted evaluation report. / 打印格式化评估报告。"""
    print("=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"  Evaluator: {report.evaluator_name}")
    print(f"  Test Set:  {report.test_set_path}")
    print(f"  Queries:   {len(report.query_results)}")
    print(f"  Time:      {report.total_elapsed_ms:.0f} ms")
    print()

    # Aggregate metrics / 聚合指标
    print("─" * 60)
    print("  AGGREGATE METRICS")
    print("─" * 60)
    if report.aggregate_metrics:
        for metric, value in sorted(report.aggregate_metrics.items()):
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"  {metric:<25s} {bar} {value:.4f}")
    else:
        print("  (no metrics computed)")
    print()

    # Per-query details / 逐查询详情
    print("─" * 60)
    print("  PER-QUERY RESULTS")
    print("─" * 60)
    for i, qr in enumerate(report.query_results, 1):
        print(f"\n  [{i}] {qr.query}")
        print(f"      Retrieved: {len(qr.retrieved_chunk_ids)} chunks")
        if qr.metrics:
            for metric, value in sorted(qr.metrics.items()):
                print(f"      {metric}: {value:.4f}")
        else:
            print("      (no metrics)")
        print(f"      Time: {qr.elapsed_ms:.0f} ms")

    print()
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
