docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)

docker run -d \
  --name my-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS=\[\"apoc\"\] \
  neo4j:latest