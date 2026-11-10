from transfer_eval.ingestion.map_tf import MapTFAdapter
from transfer_eval.ingestion.pilot import PilotAdapter
from transfer_eval.ingestion.zsc_eval import ZSCEvalAdapter
from transfer_eval.ingestion.generic import (GenericRunAdapter, DisabledAdapter, PortalAdapter,
                                             GCTMarlAdapter, BayesToMoPAdapter, EPIRLWumpusAdapter)


def make_adapter(name, spec):
    if name == "map_tf":
        return MapTFAdapter(spec)
    if name == "pilot":
        return PilotAdapter(spec)
    if name == "zsc_eval":
        return ZSCEvalAdapter(spec)
    if name == "autodrive_future":
        return DisabledAdapter(spec)
    return {
        "portal": PortalAdapter, "gct_marl": GCTMarlAdapter,
        "bayes_tomop": BayesToMoPAdapter, "epirl_wumpus": EPIRLWumpusAdapter,
    }.get(name, GenericRunAdapter)(spec)


__all__ = ["make_adapter"]
