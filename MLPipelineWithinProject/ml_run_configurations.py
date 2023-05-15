import subprocess
import gc
from ml_utils import collect_available_choices

def get_configurations():
    to_run = []
    choices = collect_available_choices()
    #these are your single projects
    choices["data"] = ["achilles","activiti", "cukes", "db-scheduler", "dnsjava", "dropwizard", "elastic-job-lite",
                       "esper", "fastjson", "hadoop", "hbase", "hutool", "java-websocket", "jfreechart", "jhipster-registry",
                       "junit-quickcheck", "noxy", "oci-java-sdk", "orbit", "otto", "retrofit", "riptide", "rxjava2-extras",
                       "sawmill", "search-highlighter", "spring-boot", "spring-cloud-zuul-ratelimit", "spring-ws",
                       "timely", "undertow", "unix4j", "vertexium", "wildfly","admiral","aismessages","alien4cloud","c2mon",
                       "carbon-apimgt","fluent-logger-java","hsac-fitnesse-fixtures","http-request","incubator-dubbo",
                       "jimfs","joda-time","luwak","marine-api","oryx","querydsl","tyrus","vertx-completable-future","webcollector",
                       "yawp","aletheia","helios","nexus-repository-helm","openpojo","pippo","struts","wikidata-toolkit","wro4j"]


    with open('configuration.txt', 'r') as f:
        for line in f.readlines():
            params = {}
            conf = line.strip().split(" ")
            # default
            for key in choices.keys():
                params[key] = "none"
            params["k"] = 10
            params["feature_sel"] = "none"

            for c in conf:
                for key in choices.keys():
                    if c in choices[key]:
                        params[key] = c
            to_run.append(params)
    return to_run
    
    
# run
for conf in get_configurations():
    subprocess.run(["python", "./ml_main.py", "-i", conf["data"], "-k", str(conf["k"]), "-p", conf["feature_sel"], conf["balancing"], conf["optimization"], conf["classifier"]])
    gc.collect()
                    

