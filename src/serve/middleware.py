import json
import logging
import uuid
from typing import Optional
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from fastapi import FastAPI, HTTPException
from starlette.types import Scope, Receive, Send, Message
from starlette.responses import RedirectResponse
import aiohttp

def get_logger():
    return logging.getLogger("ray.serve")

class OnlineStatusMiddleware(BaseHTTPMiddleware):
    OPENAI_STATUS_URL = "https://status.openai.com/api/v2/status.json"

    def __init__(
        self, 
        app: FastAPI, 
        exempt_routes: list[str] = [], 
        *args, 
        **kwargs
    ):
        super().__init__(app, *args, **kwargs)
        self.exempt_routes = exempt_routes
        # 세션을 미들웨어 초기화 시 생성하여 재사용
        self.session = aiohttp.ClientSession()

    async def __call__(
        self, 
        scope: Scope, 
        receive: Receive, 
        send: Send
    ) -> None:
        # HTTP 요청이 아닌 경우 원래 앱 호출
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        # 인증 면제 경로 체크
        if any(request.url.path.endswith(route) for route in self.exempt_routes):
            await self.app(scope, receive, send)
            return

        try:
            async with self.session.get(self.OPENAI_STATUS_URL) as response:
                if response.status == 500:
                    # 경로가 '/gemma'로 끝나지 않을 때만 리다이렉트
                    if not request.url.path.endswith('/gemma'):
                        redirect_url = request.url.path + '/gemma'
                        redirect_response = RedirectResponse(url=redirect_url)
                        await redirect_response(scope, receive, send)
                        return
                    else:
                        # 이미 '/gemma'로 끝나는 경우 원래 요청 처리
                        await self.app(scope, receive, send)
                        return
                elif response.status == 200:
                    status_data = await response.json()
                    description = status_data.get("status", {}).get("description", "")
                    if description.lower() != "all systems operational":
                        raise HTTPException(
                            status_code=503,
                            detail=f"Online Service in Error: {description}",
                        )
                else:
                    raise HTTPException(
                        status_code=503,
                        detail="Service Error...",
                    )
        except aiohttp.ClientError:
            # 상태 페이지 접근 실패 시 원래 요청 처리
            await self.app(scope, receive, send)
            return

        # 정상적인 경우 원래 요청 처리
        await self.app(scope, receive, send)

class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI 미들웨어: 요청과 응답을 로깅하는 미들웨어
    """
    logger = get_logger()

    async def dispatch(
            self, 
            request: Request, 
            call_next: RequestResponseEndpoint
        ) -> Response:
        start_time = time.time()
        self.logger.info(f"Incoming request: {request.method} {request.url.path}")
        
        # Read and parse the request body
        body = await request.body()
        try:
            if request.method == "GET":
                new_body = body
            else:
                data = json.loads(body)
                new_body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            request = Request(request.scope, receive=lambda: new_body)
            
            response = await call_next(request)
        except Exception as exc:
            self.logger.exception("Error while processing request")
            return Response("Internal Server Error", status_code=500)
                # 원본 응답의 상태를 유지하기 위해 body_iterator를 재설정
        
        process_time = time.time() - start_time        
        
        response_body = b''
        async for chunk in response.body_iterator:
            response_body += chunk
        async def new_body_iterator():
            yield response_body
        response.body_iterator = new_body_iterator()
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        self.logger.info(f"[app log] Response status code: {response.status_code}")
        self.logger.info(f"[app log] Completed {request.method} {request.url.path} in {process_time:.3f}s")
        self.logger.info(f"[app log] Request headers: {request.headers}")
        self.logger.info(f"[app log] Request body: {new_body.decode('utf-8')}")
        self.logger.info(f"[app log] Response headers: {response.headers}")
        self.logger.info(f"[app log] Response body: {response_body.decode('utf-8')}")
        return response
