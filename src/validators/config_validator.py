from typing import Any, Dict, Tuple

from jsonschema.exceptions import ValidationError
from jsonschema.validators import RefResolver
from openapi_schema_validator import OAS30Validator
from openapi_spec_validator.readers import read_from_filename


class ConfigValidator:
    def __init__(self, swagger_path: str = "api/api.yaml"):
        rooted_schemas, _ = read_from_filename(swagger_path)
        ref_resolver = RefResolver.from_schema(rooted_schemas)

        self.validator = OAS30Validator(
            rooted_schemas["components"]["schemas"]["agentConfiguration"], resolver=ref_resolver
        )
        self.ok_message = "ok"
        self.ok_status = 200
        self.error_status = 400

    def validate_agent_config(self, config: Dict[str, Any]) -> Tuple[int, str]:
        try:
            self.validator.validate(config)
        except ValidationError as e:
            mess = "Общая ошибка валидации:"
            mess += f"\n  Сообщение: {e.message}"
            mess += f"\n  Путь к данным: {list(e.path)}"
            mess += f"\n  Путь к схеме: {list(e.schema_path)}"
            mess += f"\n  Валидатор: {e.validator} (ошибка произошла для '{e.validator_value}')"

            if hasattr(e, 'context') and e.context and e.validator == 'oneOf':
                mess += "\n\n-----------------------------------------------------"
                mess += "\nДетальные ошибки для каждого 'oneOf' варианта:"
                oneof_definitions = e.validator_value

                for i, suberror in enumerate(e.context):
                    schema_name = "Неизвестная схема"
                    oneof_index = suberror.schema_path[0] if suberror.schema_path else -1
                    if 0 <= oneof_index < len(oneof_definitions):
                        schema_def = oneof_definitions[oneof_index]
                        if '$ref' in schema_def:
                            ref_path = schema_def['$ref']
                            schema_name = ref_path.split('/')[-1]
                        else:
                            schema_name = f"Встроенная схема (индекс {oneof_index})"

                    mess += f"\n--- Ошибка #{i+1} для схемы '{schema_name}' (индекс {oneof_index}) ---"
                    mess += f"\n  Сообщение: {suberror.message}"
                    mess += f"\n  Путь к данным: {list(suberror.path)}"
                    mess += f"\n  Путь к схеме: {list(suberror.schema_path)}"
                    mess += f"\n  Валидатор: {suberror.validator}"

            return self.error_status, mess

        return self.ok_status, self.ok_message
