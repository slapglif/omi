#!/bin/bash
# Test script for shared namespace REST API endpoints
# Usage: ./test_namespace_api.sh

set -e

BASE_URL="http://localhost:8420"
API_KEY="${OMI_API_KEY:-development}"

echo "Testing Shared Namespace REST API Endpoints"
echo "============================================"
echo

# Test 1: Create namespace
echo "1. Creating shared namespace..."
CREATE_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/namespaces/shared" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"namespace": "test/shared", "created_by": "test-agent"}' \
  -w "\nHTTP_CODE:%{http_code}")

HTTP_CODE=$(echo "$CREATE_RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
RESPONSE_BODY=$(echo "$CREATE_RESPONSE" | sed '/HTTP_CODE:/d')

echo "Status: $HTTP_CODE"
echo "Response: $RESPONSE_BODY"
echo

if [ "$HTTP_CODE" = "201" ]; then
    echo "✓ Create namespace passed"
else
    echo "✗ Create namespace failed (expected 201, got $HTTP_CODE)"
fi
echo

# Test 2: List namespaces
echo "2. Listing shared namespaces..."
LIST_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/namespaces/shared" \
  -H "X-API-Key: $API_KEY" \
  -w "\nHTTP_CODE:%{http_code}")

HTTP_CODE=$(echo "$LIST_RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
RESPONSE_BODY=$(echo "$LIST_RESPONSE" | sed '/HTTP_CODE:/d')

echo "Status: $HTTP_CODE"
echo "Response: $RESPONSE_BODY"
echo

if [ "$HTTP_CODE" = "200" ]; then
    echo "✓ List namespaces passed"
else
    echo "✗ List namespaces failed (expected 200, got $HTTP_CODE)"
fi
echo

# Test 3: Get namespace
echo "3. Getting namespace info..."
GET_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/namespaces/shared/test%2Fshared" \
  -H "X-API-Key: $API_KEY" \
  -w "\nHTTP_CODE:%{http_code}")

HTTP_CODE=$(echo "$GET_RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
RESPONSE_BODY=$(echo "$GET_RESPONSE" | sed '/HTTP_CODE:/d')

echo "Status: $HTTP_CODE"
echo "Response: $RESPONSE_BODY"
echo

if [ "$HTTP_CODE" = "200" ]; then
    echo "✓ Get namespace passed"
else
    echo "✗ Get namespace failed (expected 200, got $HTTP_CODE)"
fi
echo

# Test 4: Grant permission
echo "4. Granting permission..."
GRANT_RESPONSE=$(curl -s -X POST "$BASE_URL/api/v1/namespaces/shared/test%2Fshared/permissions" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"agent_id": "test-agent", "target_agent_id": "another-agent", "permission_level": "read"}' \
  -w "\nHTTP_CODE:%{http_code}")

HTTP_CODE=$(echo "$GRANT_RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
RESPONSE_BODY=$(echo "$GRANT_RESPONSE" | sed '/HTTP_CODE:/d')

echo "Status: $HTTP_CODE"
echo "Response: $RESPONSE_BODY"
echo

# Test 5: List permissions
echo "5. Listing permissions..."
LIST_PERM_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/namespaces/shared/test%2Fshared/permissions" \
  -H "X-API-Key: $API_KEY" \
  -w "\nHTTP_CODE:%{http_code}")

HTTP_CODE=$(echo "$LIST_PERM_RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
RESPONSE_BODY=$(echo "$LIST_PERM_RESPONSE" | sed '/HTTP_CODE:/d')

echo "Status: $HTTP_CODE"
echo "Response: $RESPONSE_BODY"
echo

if [ "$HTTP_CODE" = "200" ]; then
    echo "✓ List permissions passed"
else
    echo "✗ List permissions failed (expected 200, got $HTTP_CODE)"
fi
echo

# Test 6: Check permission
echo "6. Checking permission..."
CHECK_RESPONSE=$(curl -s -X GET "$BASE_URL/api/v1/namespaces/shared/test%2Fshared/permissions/test-agent/check" \
  -H "X-API-Key: $API_KEY" \
  -w "\nHTTP_CODE:%{http_code}")

HTTP_CODE=$(echo "$CHECK_RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
RESPONSE_BODY=$(echo "$CHECK_RESPONSE" | sed '/HTTP_CODE:/d')

echo "Status: $HTTP_CODE"
echo "Response: $RESPONSE_BODY"
echo

if [ "$HTTP_CODE" = "200" ]; then
    echo "✓ Check permission passed"
else
    echo "✗ Check permission failed (expected 200, got $HTTP_CODE)"
fi
echo

echo "============================================"
echo "Test suite completed"
echo
echo "Note: To clean up, you can delete the test namespace using:"
echo "curl -X DELETE '$BASE_URL/api/v1/namespaces/shared/test%2Fshared?agent_id=test-agent' -H 'X-API-Key: $API_KEY'"
