from neo4j import GraphDatabase
from transformers import pipeline
import spacy, coreferee
from transformers import AutoTokenizer
import pprint

username = "neo4j"
password = "password"
url = "bolt://localhost:7687"

driver = GraphDatabase.driver(url, auth=(username, password))


coref_nlp = spacy.load("en_core_web_lg")
coref_nlp.add_pipe("coreferee")

tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")


def coref_text(text):
    coref_doc = coref_nlp(text)
    resolved_text = ""

    for token in coref_doc:
        repres = coref_doc._.coref_chains.resolve(token)
        if repres:
            resolved_text += " " + " and ".join(
                [
                    (
                        t.text
                        if t.ent_type_ == ""
                        else [e.text for e in coref_doc.ents if t in e][0]
                    )
                    for t in repres
                ]
            )
        else:
            resolved_text += " " + token.text

    return resolved_text


def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    for token in (
        text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
    ):
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject != "" and relation != "" and object_ != "":
        triplets.append(
            {"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()}
        )
    return triplets


def save_triplets_to_neo4j(driver, triplets):
    with driver.session() as session:
        # Clear existing graph
        session.run("MATCH (n) DETACH DELETE n")

        # Save each triplet
        for triplet in triplets:
            head = triplet.get("head")
            relation = triplet.get("type")
            tail = triplet.get("tail")

            if head and relation and tail:
                cypher = """
                MERGE (s:Entity {name: $head})
                MERGE (o:Entity {name: $tail})
                MERGE (s)-[:RELATION {type: $relation}]->(o)
                """
                session.run(cypher, head=head, relation=relation, tail=tail)


def split_text(text, tokenizer, max_tokens=600, stride=200):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start += max_tokens - stride  # overlap
    return chunks


def dedup_triplets(triplets):

    unique = list({(item["head"], item["type"], item["tail"]) for item in triplets})
    deduped = [{"head": h, "type": t, "tail": tail} for (h, t, tail) in unique]

    return deduped


if __name__ == "__main__":
    with open("input2.txt") as file:
        text = file.read()

    # resolve he / she to objects with spacy lib 'en_core_web_lg' model
    text_corefed = coref_text(text)

    triplet_extractor = pipeline(
        "text2text-generation",
        model="Babelscape/rebel-large",
        tokenizer="Babelscape/rebel-large",
    )

    triplets = []
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    for chunk in split_text(text_corefed, tokenizer):
        extracted_text = triplet_extractor.tokenizer.batch_decode(
            [
                triplet_extractor(chunk, return_tensors=True, return_text=False)[0][
                    "generated_token_ids"
                ]
            ]
        )
        triplet = extract_triplets(extracted_text[0])
        triplets += triplet

    triplets = dedup_triplets(triplets)
    pprint.pprint(triplets)

    # Save triplets to Neo4j
    save_triplets_to_neo4j(driver, triplets)
