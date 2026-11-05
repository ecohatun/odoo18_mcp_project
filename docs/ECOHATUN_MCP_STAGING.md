# SMART ECOHATUN · Incremento de lectura para MCP Odoo 18

**COMPROBADO localmente.** Este incremento añade `ecohatun_mcp.py` a la raíz de `ecohatun/odoo18_mcp_project`. Reutiliza el cliente XML-RPC del repositorio. El punto de entrada separado permite probar el cambio en staging antes de cambiar consumidores. El contrato REST legado y sus adaptadores requieren una revisión de compatibilidad.

## Instalación de prueba

1. En una copia aislada del repositorio, copia `ecohatun_mcp.py` a la raíz. Instala las dependencias del proyecto y resuelve el SDK con `mcp==1.26.0`, que fue la versión probada localmente.
2. Configura `ODOO_URL` y `ODOO_DB` para Odoo de staging. Define cuentas Odoo de lectura distintas con ACL y reglas de registros verificadas. Usa el almacén de secretos para cada `ODOO_MCP_{PERFIL}_TOKEN` y `ODOO_MCP_{PERFIL}_API_KEY`, y configura su usuario en `ODOO_MCP_{PERFIL}_USER`. Genera cada token de forma independiente con `secrets.token_urlsafe(32)`.
3. Define `ODOO_MCP_MODEL_FIELDS` como JSON. Ejemplo de estructura: `{"comercial":{"product.template":["name"]},"interno":{"sale.order":["state","amount_total","date_order"]}}`. Ajusta empresas, modelos y campos a la matriz aprobada. En Odoo, comprueba que cada cuenta solo alcance sus empresas y registros autorizados.
4. Define `ODOO_MCP_ALLOWED_HOSTS` como lista de cabeceras Host exactas separadas por comas. En local, prueba con `python -m uvicorn ecohatun_mcp:create_app --factory --host 127.0.0.1 --port 8000`. El endpoint MCP es `/mcp`, con token Bearer del perfil.
5. Para repetir la prueba local, ejecuta `PYTHONPATH=. python tests/check_ecohatun_mcp.py` desde la raíz del repositorio. La prueba utiliza credenciales sintéticas y sustituye la llamada a Odoo en memoria.

## Qué aporta

- Transporte MCP Streamable HTTP, autenticación por perfil y comprobación de Host.
- `search_records`, `search_count` y `aggregate_records` con listas de modelos y campos permitidos.
- Agregación mediante el método fijo `read_group` de Odoo 18. El cliente MCP elige campos autorizados, no métodos ni SQL.
- Filtros AND acotados sin recorridos de relaciones, proyección de los resultados y límite de filas para listados y agrupaciones.

## Evidencia y fase siguiente

En el VPS, dos archivos de `odoodo-mcp` coinciden con el repositorio Ecohatun. Tres archivos de `odoo18-mcp-hostinger` coinciden con ERPipe, y su imagen monta un guard HTTP local. El adaptador Node monta `/app/server.cjs` y contiene despacho `tools/list` y `tools/call`.

La prueba local cubrió inicialización MCP, catálogo, autenticación, Host, rechazo por modelo, campo, filtro y límite, y proyección del resultado simulado. Quedan pendientes la resolución de dependencias dentro de la imagen del repositorio, las pruebas en Odoo 18 de staging con dos empresas, los clientes reales, el consumo y la compatibilidad de los adaptadores. Las escrituras aprobadas, la auditoría de negocios y RAG SQL son incrementos posteriores. La reversión del piloto consiste en conservar el servicio anterior como destino de los consumidores hasta aprobar el cambio.


## Catálogo y patrones versionados

- `list_allowed_models` enumera los modelos configurados para el perfil. `describe_model` consulta `fields_get` con la cuenta Odoo del perfil y entrega solo etiquetas y tipos de campos permitidos.
- `list_saved_queries` y `run_saved_query` exponen un patrón versionado de conteos por estado. Los parámetros son fechas UTC, con inicio incluido, fin excluido y hasta 366 días. Usa el ORM y consulta los datos actuales. Requiere acceso a `sale.order.state` y `sale.order.date_order`.
- Los patrones se incorporan mediante revisión de código y pruebas. No hay publicación automática de consultas generadas ni persistencia de resultados. RAG SQL queda pendiente para consultas que excedan las herramientas del ORM y requiere un validador y acceso de solo lectura independientes.
- Prueba local adicional: metadatos proyectados, rechazo por perfil y rango, y forma de los parámetros enviados al ORM simulado. Pendiente validar en Odoo 18 de staging.
