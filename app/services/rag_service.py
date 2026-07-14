from app.models import (
    ChatResponse,ResponseMetadata,
    RetrievedChunk,RetrievedChunkPreview
)

from app.security.spotlighting import build_spotlighted_context
from app.security.system_prompt import build_system_prompt
from app.services.embedding_service import embed_texts
from app.services.llm_service import generate
from app.services.vector_store import search,hybrid_search,sparse_search
from loguru import logger 
from app.config import settings
from app.services.reranking import Reranker
from app.services.hyde import HyDERetriever
from app.services.crag import crag_pipeline
from app.services.self_reflective import reflect_on_answer,should_regenerate

  


def _flag(flags:dict |None,key:str,default):
    if not isinstance(flags,dict):
        return default
    return flags.get(key,default)

def _retrieve(question:str , flags:dict | int |None=None)-> list[RetrievedChunk]:
    final_top_k = int(_flag(flags,"top_k",5))
    mode=_flag(flags,"search_mode","dense")
    rerank=bool(_flag(flags,"enable_rerank",False))
    hyde=bool(_flag(flags,"enable_hyde",False))
    retrieve_k=settings.reranker_initial_top_k if rerank else final_top_k
    enable_crag=bool(_flag(flags,"enable_crag",settings.crag_enabled_by_default))


    if hyde:
        chunks=HyDERetriever().retrieve(question,top_k=retrieve_k)    


    elif mode=="sparse":
        return sparse_search(question,top_k=retrieve_k)
    elif mode=="hybrid":
        query_embedding= embed_texts([question])[0]
        chunks=hybrid_search(query_embedding,question,top_k=retrieve_k)
    else:
        query_embedding= embed_texts([question])[0]
        chunks=search(query_embedding,top_k=retrieve_k) # DENSE search is the default search mode, which uses vector embeddings to find the most relevant chunks based on semantic similarity to the query.

    if rerank and chunks:
        chunks=Reranker().rerank(question,chunks,top_k=final_top_k)
    else:
        chunks=chunks[:final_top_k]
          # Ensure we only return the top_k chunks if reranking is not enabled 
    chunks,evaluation,used_web=crag_pipeline(question=question,chunks=chunks,enable_crag=enable_crag)
    logger.info("CRAG | enabled={} score={}  label={}, used_web={}",enable_crag,evaluation.relevance_score,evaluation.relevance_label,used_web)
    return chunks   


def _generate(
    question:str,
    chunks:list[RetrievedChunk],
    flags:dict|None=None,)-> ChatResponse:
    enable_self_reflective=bool(_flag(flags,"enable_self_reflective",False))
    spotlighted=build_spotlighted_context(chunks)
    system=build_system_prompt()
    def _raw(q:str)->str:
        return generate(system,f"{spotlighted}\n\nQuestion:{q}")["text"]

    working_q=question
    raw=_raw(working_q) # NORMAL GENERATION WITHOUT SELF REFLECTION

    # self-rag:reflect on the answer: refine the question and retry if weak
    iterations=0
    last_score:float|None=None
    final_refined:str|None=None
    if enable_self_reflective:
        while True:
            reflection=reflect_on_answer(
                question=working_q,
                answer=raw,
                context=spotlighted
            )
            last_score=float(reflection.reflection_score)
            if not should_regenerate(reflection,iterations):
                break
            final_refined=reflection.refined_question or working_q
            working_q=final_refined
            raw=_raw(working_q)
            iterations+=1

    chunk_previews=[
        RetrievedChunkPreview(text=c.text,source=c.source,score=c.score) for c in chunks 
    ]
    return ChatResponse(
        answer=raw,
        sources=list({c.source for c in chunks}),
        confidence=0.7,# BECAUSE THE THRESHOLD IS 0.7
        metadata=ResponseMetadata(
            route="rag",
            retrieved_chunks=chunk_previews,
            reflection_iterations=iterations,
            refleciton_score=last_score,
            refined_question=final_refined
        )

    )           






     

def run_rag(question:str,flags:dict | int |None=None)->ChatResponse:
    logger.info("L5 RAG | search_mode={} rerank={} hyde={} crag={} top_k={}",_flag(flags,"search_mode","dense"),
    _flag(flags,"enable_rerank",False),
    _flag(flags,"enable_hyde",False),_flag
    (flags,"enable_crag",settings.crag_enabled_by_default),_flag(flags,"enable_self_reflective",False),_flag(flags,"top_k",5))
    flags_dict=flags if isinstance(flags,dict)else None
    chunks=_retrieve(question,flags=flags_dict)
    return _generate(question,chunks,flags=flags_dict)


def run_rag_with_trace(question:str,flags:dict | int |None=None)->tuple[ChatResponse,list[RetrievedChunk]]:
    # this function is for evaluation purposes, it returns the retrieved chunks along with the response

    chunks=_retrieve(question,flags=flags if isinstance(flags,dict) else None)
    response= _generate(question,chunks,flags=flags if isinstance(flags,dict)else None)
    return response,chunks


run_rag_with_trace_no_cache=run_rag_with_trace    

  