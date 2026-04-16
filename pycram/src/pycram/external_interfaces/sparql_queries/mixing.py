from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper(
    "https://knowledgedb.informatik.uni-bremen.de/mealprepDB/MealPreparation/query"
)
sparql.setReturnFormat(JSON)

prefix = """
 PREFIX owl: <http://www.w3.org/2002/07/owl#>
 PREFIX cut: <http://www.ease-crc.org/ont/meals#>
 PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
 PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
 PREFIX foodon: <http://purl.obolibrary.org/obo/>
 PREFIX soma: <http://www.ease-crc.org/ont/SOMA.owl#>
 PREFIX sit_aware: <http://www.ease-crc.org/ont/situation_awareness#>
 """


def get_motion(task):
    query = """
    SELECT  ?motion WHERE {
  cut:%s rdfs:subClassOf ?restriction .
  ?restriction owl:onProperty cut:requiresMotion.
  ?restriction owl:someValuesFrom ?motion .
} """ % (
        task
    )
    full_query = prefix + query
    sparql.setQuery(full_query)
    results = sparql.queryAndConvert()
    return (
        results["results"]["bindings"][0]["motion"]["value"]
        if results["results"]["bindings"]
        else None
    )


def get_mixing_tool(task):
    query = """
        SELECT  ?tool WHERE {
      cut:%s rdfs:subClassOf ?restriction .
      ?restriction owl:onProperty soma:affordsTrigger.
      ?restriction owl:someValuesFrom ?tool .
    } """ % (
        task
    )
    full_query = prefix + query
    sparql.setQuery(full_query)
    results = sparql.queryAndConvert()
    return (
        results["results"]["bindings"][0]["tool"]["value"]
        if results["results"]["bindings"]
        else None
    )


def get_mixing_knowledge(task):
    return {
        "query_success": True,
        "task": task,
        "motion": get_motion(task),
        "mixing_tool": get_mixing_tool(task),
    }


def safe_get_mixing_knowledge(task):
    try:
        return get_mixing_knowledge(task)
    except Exception as exc:
        return {
            "query_success": False,
            "task": task,
            "motion": None,
            "mixing_tool": None,
            "query_error": f"{type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    print(get_motion("Folding"))
    print(get_mixing_tool("Whisking"))
