from collections import Counter
from pathlib import Path
from threading import Barrier, Lock

import pytest

from mapcl.io import read_table
from mapcl.llm import Assessment, Expansion
from mapcl.mapper import CellOntologyMapper
from mapcl.ols import Candidate, OLSError

from .conftest import FakeLLM, FakeOLS, cl


@pytest.mark.parametrize(("workers", "ols_workers"), [(4, 1), (4, 2), (10, 5)])
def test_llm_calls_overlap_independently_of_ols_limit(ontology, workers, ols_workers):
    lock = Lock()
    ols_barrier = Barrier(ols_workers, timeout=5)
    llm_barrier = Barrier(workers, timeout=5)
    active = Counter()
    peak = Counter()

    def enter(kind):
        with lock:
            active[kind] += 1
            peak[kind] = max(peak[kind], active[kind])

    def leave(kind):
        with lock:
            active[kind] -= 1

    class BlockingOLS(FakeOLS):
        def search(self, query, ontologies=("cl",), rows=5):
            enter("ols")
            try:
                ols_barrier.wait()
                return [cl("CL:0000235", "macrophage")]
            finally:
                leave("ols")

    def respond(schema, prompt):
        enter("llm")
        try:
            llm_barrier.wait()
            return Assessment(best_index=1, score=9, reason="test match")
        finally:
            leave("llm")

    mapper = CellOntologyMapper(FakeLLM(respond), ontology=ontology, ols=BlockingOLS(),
                                max_concurrency=workers, ols_concurrency=ols_workers)
    labels = [f"mystery {chr(97 + i)}" for i in range(workers)]
    results = mapper.map(labels)
    assert [r.input_label for r in results] == labels
    assert all(r.method == "llm" for r in results)
    assert peak == {"ols": ols_workers, "llm": workers}


def test_ols_slot_is_released_after_error(ontology):
    class FailOnceOLS(FakeOLS):
        def search(self, query, ontologies=("cl",), rows=5):
            if not self.calls:
                self.calls.append((query, tuple(ontologies), rows))
                raise OLSError("test failure")
            return []

    mapper = CellOntologyMapper(ontology=ontology, ols=FailOnceOLS(), ols_concurrency=1)
    with pytest.raises(OLSError, match="test failure"):
        mapper.map_one("mystery")
    assert mapper._ols_slots.acquire(blocking=False)
    mapper._ols_slots.release()
    assert mapper.map_one("mystery").method == "unmapped"


@pytest.mark.parametrize("kwargs", [
    {"max_concurrency": 0}, {"max_concurrency": -1},
    {"ols_concurrency": 0}, {"ols_concurrency": -1},
])
def test_concurrency_limits_must_be_positive(ontology, kwargs):
    with pytest.raises(ValueError, match="must be positive"):
        CellOntologyMapper(ontology=ontology, ols=FakeOLS(), **kwargs)


def test_example_results_and_requests_match_serial_execution():
    """Exercise the real example labels with deterministic offline responses."""
    source = Path(__file__).resolve().parents[1] / "examples" / "annotated_cells_example.tsv"
    table = read_table(source)
    names = dict(zip(table["cell type"], table["cell_type_catelog"]))

    class ExampleOLS(FakeOLS):
        def search(self, query, ontologies=("cl",), rows=5):
            self.calls.append((query, tuple(ontologies), rows))
            if "cl" in ontologies:
                return [cl("CL:0000235", "macrophage")]
            return [Candidate("NCIT:C123", "test fallback cell", "NCIT")]

    def respond(schema, prompt):
        if schema is Expansion:
            label = prompt.split("Label: ", 1)[1].splitlines()[0]
            return Expansion(cell_type=names[label], synonyms=[])
        # Reject some CL rounds to exercise late expansion and ontology fallback.
        label = prompt.split("User label: ", 1)[1].splitlines()[0]
        score = 4 if sum(map(ord, label)) % 3 == 0 and "NCIT:C123" not in prompt else 9
        return Assessment(best_index=1, score=score, reason="offline regression test")

    def run(workers):
        llm, ols = FakeLLM(respond), ExampleOLS()
        mapper = CellOntologyMapper(llm, expand=True, max_concurrency=workers,
                                    ols_concurrency=5, ols=ols)
        results = mapper.map(table["cell type"])
        assert all(r.method != "error" for r in results)
        return results, Counter(llm.prompts), Counter(ols.calls)

    serial = run(1)
    parallel = run(10)
    assert parallel == serial
    assert {r.method for r in parallel[0]} >= {"direct", "llm", "llm_fallback"}
