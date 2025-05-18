import json
import yaml

# Adjusted for running inside backend/
input_path = "project-echo-openapi.json"
output_path = "project-echo-openapi.yaml"

with open(input_path, "r") as json_file:
    openapi_data = json.load(json_file)

with open(output_path, "w") as yaml_file:
    yaml.dump(openapi_data, yaml_file, sort_keys=False)

print(f"✅ Converted OpenAPI to YAML: {output_path}")
