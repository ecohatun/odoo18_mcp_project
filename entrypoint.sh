#!/bin/sh

# Create required directories with proper permissions
mkdir -p /app/logs /app/data /app/exports /app/tmp /app/generated_modules
chown -R mcp:mcp /app/logs /app/data /app/exports /app/tmp /app/generated_modules

# Switch to non-root user and execute
exec su -s /bin/sh mcp -c "
if [ \"\$1\" = \"test\" ]; then
    if [ \"\$2\" = \"mcp\" ]; then
        exec python tests/test_mcp_server_consolidated.py --all
    elif [ \"\$2\" = \"agent\" ]; then
        exec python tests/test_odoo_code_agent_consolidated.py --all
    elif [ \"\$2\" = \"utils\" ]; then
        exec python tests/test_odoo_code_agent_utils_consolidated.py --all
    elif [ \"\$2\" = \"export-import\" ]; then
        exec python tests/test_export_import_agent.py
    elif [ \"\$2\" = \"all\" ]; then
        python tests/test_mcp_server_consolidated.py --all && \
        python tests/test_odoo_code_agent_consolidated.py --all && \
        python tests/test_odoo_code_agent_utils_consolidated.py --all && \
        python tests/test_export_import_agent.py
    else
        echo \"Unknown test type: \$2\"
        echo \"Available test types: mcp, agent, utils, export-import, all\"
        exit 1
    fi
elif [ \"\$1\" = \"main\" ]; then
    exec python main.py \${@:2}
else
    # Por defecto ejecutar standalone_mcp_server.py
    exec python standalone_mcp_server.py
fi
"
