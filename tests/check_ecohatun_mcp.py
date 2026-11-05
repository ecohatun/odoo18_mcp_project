import json, os
from unittest.mock import patch
from starlette.testclient import TestClient
import ecohatun_mcp as app

os.environ.update({
 "ODOO_MCP_MODEL_FIELDS": json.dumps({"comercial":{"product.template":["name"]},
                                      "interno":{"sale.order":["state","amount_total","date_order"]}}),
 "ODOO_MCP_COMERCIAL_TOKEN":"A"*40,"ODOO_MCP_COMERCIAL_USER":"service-commercial",
 "ODOO_MCP_COMERCIAL_API_KEY":"synthetic","ODOO_MCP_INTERNO_TOKEN":"B"*40,
 "ODOO_MCP_INTERNO_USER":"service-internal","ODOO_MCP_INTERNO_API_KEY":"synthetic",
})
with patch.object(app, "odoo_read") as rpc, TestClient(app.create_app(),base_url="http://localhost") as client:
 def request(method,token=None,params=None,host=None):
    headers={"Accept":"application/json, text/event-stream"}
    if token: headers["Authorization"]="Bearer "+token
    if host: headers["Host"]=host
    return client.post("/mcp",headers=headers,json={"jsonrpc":"2.0","id":1,"method":method,"params":params or {}})
 def tool(name,token,**args):
    return request("tools/call",token,{"name":name,"arguments":args}).json()["result"]
 assert request("tools/list").status_code==401
 assert request("tools/list","Z"*40).status_code==401
 assert request("tools/list","A"*40,host="other.invalid").status_code in (400,403,421)
 init=request("initialize","A"*40,{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"1"}})
 assert init.status_code==200 and "result" in init.json(), init.text
 names={x["name"] for x in request("tools/list","A"*40).json()["result"]["tools"]}
 assert names=={"search_records","search_count","aggregate_records","list_allowed_models","describe_model","list_saved_queries","run_saved_query"},names
 for args in ({"model":"sale.order","fields":["state"]},{"model":"product.template","fields":["list_price"]},
              {"model":"product.template","fields":["name"],"domain":[["partner_id.email","=","x"]]},
              {"model":"product.template","fields":["name"],"limit":101}):
    assert tool("search_records","A"*40,**args)["isError"]
 assert not rpc.called
 rpc.return_value=[{"state":"sale","record_count":2,"total":19,"__domain":[]}]
 r=tool("aggregate_records","B"*40,model="sale.order",group_by="state",sum_field="amount_total")
 assert not r.get("isError"),r
 assert "__domain" not in str(r)
 assert rpc.call_args.args[1]=="read_group"
 assert rpc.call_args.args[2][1]==["state","record_count:count(id)","total:sum(amount_total)"]
 rpc.return_value=[{"name":"Termas","id":1}]
 r=tool("search_records","A"*40,model="product.template",fields=["name"])
 assert not r.get("isError") and '"id"' not in str(r),r
 rpc.reset_mock()
 assert tool("describe_model","A"*40,model="sale.order")["isError"]
 args={"query_id":"sale_orders_by_status.v1","start":"2026-01-01","end":"2026-02-01"}
 assert tool("run_saved_query","A"*40,**args)["isError"]
 assert tool("run_saved_query","B"*40,**{**args,"end":"2028-01-01"})["isError"]
 assert not rpc.called
 rpc.return_value={"name":{"string":"Name","type":"char","secret":"x"},"list_price":{"string":"Price"}}
 r=tool("describe_model","A"*40,model="product.template")
 assert not r.get("isError") and "list_price" not in str(r) and "secret" not in str(r)
 assert rpc.call_args.args[1:3]==("fields_get",[["name"]])
 rpc.return_value=[{"state":"sale","record_count":2}]
 assert not tool("run_saved_query","B"*40,**args).get("isError")
 assert rpc.call_args.args[1]=="read_group"
 assert rpc.call_args.args[2][0]==[["date_order",">=","2026-01-01 00:00:00"],
                                  ["date_order","<","2026-02-01 00:00:00"]]
 print("PASS: MCP initialize/tools/list, credential rejection, Host allowlist, role/model/field/domain/limit checks, read_group routing and projected results")
