"""EvaluationHarness: golden-query testing for the whole pipeline.

Each GoldenQuery pairs a realistic free-text request with the catalog songs
a human judged to be correct answers. The harness runs every query through
the full pipeline and measures:

  - recall@k    — how many of the expected songs made the top k
  - precision@k — how much of the top k was expected
  - hallucinations — recommended titles that don't exist in the catalog
                     (should ALWAYS be 0 thanks to the Validator)

Run it directly:  python -m src.evaluation
"""

from dataclasses import dataclass, field
from typing import List

from src.pipeline import RecommenderPipeline


@dataclass
class GoldenQuery:
    query: str
    expected_titles: List[str]


GOLDEN_QUERIES = [
    GoldenQuery(
        "calm acoustic songs for studying late at night",
        ["Library Rain", "Midnight Coding", "Focus Flow", "Spacewalk Thoughts"],
    ),
    GoldenQuery(
        "high energy workout music",
        ["Neon Warehouse", "Gym Hero", "Storm Runner", "Iron Verdict"],
    ),
    GoldenQuery(
        "happy upbeat pop songs",
        ["Sunrise City", "Rooftop Lights", "Island Time"],
    ),
    GoldenQuery(
        "dark intense metal",
        ["Iron Verdict", "Storm Runner"],
    ),
    GoldenQuery(
        "romantic r&b for a date night",
        ["Velvet Hours"],
    ),
    GoldenQuery(
        "sad slow classical music",
        ["Autumn Nocturne"],
    ),
]


@dataclass
class QueryResult:
    query: str
    expected: List[str]
    got: List[str]
    recall_at_k: float
    precision_at_k: float
    source: str


@dataclass
class EvalReport:
    k: int
    results: List[QueryResult] = field(default_factory=list)
    hallucination_count: int = 0

    @property
    def avg_recall_at_k(self) -> float:
        return sum(r.recall_at_k for r in self.results) / len(self.results)

    @property
    def avg_precision_at_k(self) -> float:
        return sum(r.precision_at_k for r in self.results) / len(self.results)

    def to_markdown(self) -> str:
        lines = [
            f"| Query | Expected in top {self.k} | Recall@{self.k} | Source |",
            "|---|---|---|---|",
        ]
        for r in self.results:
            hits = sorted(set(r.expected) & set(r.got))
            lines.append(
                f"| {r.query} | {len(hits)}/{len(r.expected)} "
                f"({', '.join(hits) if hits else 'none'}) "
                f"| {r.recall_at_k:.2f} | {r.source} |"
            )
        lines.append("")
        lines.append(f"**Average recall@{self.k}: {self.avg_recall_at_k:.2f}** · "
                     f"average precision@{self.k}: {self.avg_precision_at_k:.2f} · "
                     f"hallucinated titles: {self.hallucination_count}")
        return "\n".join(lines)


class EvaluationHarness:
    def __init__(self, golden_queries: List[GoldenQuery] = None, k: int = 5):
        self.golden_queries = golden_queries or GOLDEN_QUERIES
        self.k = k

    def run(self, pipeline: RecommenderPipeline) -> EvalReport:
        catalog_titles = {s.title for s in pipeline.retriever.index.songs}
        report = EvalReport(k=self.k)

        for gq in self.golden_queries:
            recs = pipeline.recommend(gq.query, k=self.k)
            got = [r.song.title for r in recs]
            hits = set(gq.expected_titles) & set(got)
            report.results.append(QueryResult(
                query=gq.query,
                expected=gq.expected_titles,
                got=got,
                recall_at_k=len(hits) / len(gq.expected_titles),
                precision_at_k=len(hits) / self.k,
                source=pipeline.last_source,
            ))
            report.hallucination_count += sum(
                1 for title in got if title not in catalog_titles
            )

        return report


def main() -> None:
    pipeline = RecommenderPipeline.from_catalog()
    mode = "gemini" if pipeline.generator.is_available() else "fallback (no API key)"
    print(f"Running golden-query evaluation in {mode} mode...\n")
    report = EvaluationHarness().run(pipeline)
    print(report.to_markdown())


if __name__ == "__main__":
    main()
