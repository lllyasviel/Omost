"""
This file defines a useful high-level abstraction to build Gradio chatbots: ChatInterface.
"""

from __future__ import annotations

import inspect
from typing import AsyncGenerator, Callable, Literal, Union, cast

import anyio
from gradio_client.documentation import document

from gradio.blocks import Blocks
from gradio.components import (
    Button,
    Chatbot,
    Component,
    Markdown,
    MultimodalTextbox,
    State,
    Textbox,
    get_component_instance,
    Dataset,
)
from gradio.events import Dependency, on
from gradio.helpers import special_args
from gradio.layouts import Accordion, Group, Row
from gradio.routes import Request
from gradio.themes import ThemeClass as Theme
from gradio.utils import SyncToAsyncIterator, async_iteration, async_lambda


@document()
class ChatInterface(Blocks):
    """
    ChatInterface is Gradio's high-level abstraction for creating chatbot UIs, and allows you to create
    a web-based demo around a chatbot model in a few lines of code. Only one parameter is required: fn, which
    takes a function that governs the response of the chatbot based on the user input and chat history. Additional
    parameters can be used to control the appearance and behavior of the demo.

    Example:
        import gradio as gr

        def echo(message, history):
            return message

        demo = gr.ChatInterface(fn=echo, examples=["hello", "hola", "merhaba"], title="Echo Bot")
        demo.launch()
    Demos: chatinterface_multimodal, chatinterface_random_response, chatinterface_streaming_echo
    Guides: creating-a-chatbot-fast, sharing-your-app
    """

    def __init__(
        self,
        fn: Callable,
        post_fn: Callable,
        pre_fn: Callable,
        chatbot: Chatbot,
        *,
        post_fn_kwargs: dict = None,
        pre_fn_kwargs: dict = None,
        multimodal: bool = False,
        textbox: Textbox | MultimodalTextbox | None = None,
        additional_inputs: str | Component | list[str | Component] | None = None,
        additional_inputs_accordion_name: str | None = None,
        additional_inputs_accordion: str | Accordion | None = None,
        examples: Dataset = None,
        title: str | None = None,
        description: str | None = None,
        theme: Theme | str | None = None,
        css: str | None = None,
        js: str | None = None,
        head: str | None = None,
        analytics_enabled: bool | None = None,
        submit_btn: str | None | Button = "Submit",
        stop_btn: str | None | Button = "Stop",
        retry_btn: str | None | Button = "🔄  Retry",
        undo_btn: str | None | Button = "↩️ Undo",
        clear_btn: str | None | Button = "🗑️  Clear",
        autofocus: bool = True,
        concurrency_limit: int | None | Literal["default"] = "default",
        fill_height: bool = True,
        delete_cache: tuple[int, int] | None = None,
    ):
        super().__init__(
            analytics_enabled=analytics_enabled,
            mode="chat_interface",
            css=css,
            title=title or "Gradio",
            theme=theme,
            js=js,
            head=head,
            fill_height=fill_height,
            delete_cache=delete_cache,
        )

        if post_fn_kwargs is None:
            post_fn_kwargs = []

        self.post_fn = post_fn
        self.post_fn_kwargs = post_fn_kwargs

        self.pre_fn = pre_fn
        self.pre_fn_kwargs = pre_fn_kwargs

        self.interrupter = State(None)

        self.multimodal = multimodal
        self.concurrency_limit = concurrency_limit
        self.fn = fn
        self.is_async = inspect.iscoroutinefunction(
            self.fn
        ) or inspect.isasyncgenfunction(self.fn)
        self.is_generator = inspect.isgeneratorfunction(
            self.fn
        ) or inspect.isasyncgenfunction(self.fn)

        if additional_inputs:
            if not isinstance(additional_inputs, list):
                additional_inputs = [additional_inputs]
            self.additional_inputs = [
                get_component_instance(i)
                for i in additional_inputs  # type: ignore
            ]
        else:
            self.additional_inputs = []
        if additional_inputs_accordion_name is not None:
            print(
                "The `additional_inputs_accordion_name` parameter is deprecated and will be removed in a future version of Gradio. Use the `additional_inputs_accordion` parameter instead."
            )
            self.additional_inputs_accordion_params = {
                "label": additional_inputs_accordion_name
            }
        if additional_inputs_accordion is None:
            self.additional_inputs_accordion_params = {
                "label": "Additional Inputs",
                "open": False,
            }
        elif isinstance(additional_inputs_accordion, str):
            self.additional_inputs_accordion_params = {
                "label": additional_inputs_accordion
            }
        elif isinstance(additional_inputs_accordion, Accordion):
            self.additional_inputs_accordion_params = (
                additional_inputs_accordion.recover_kwargs(
                    additional_inputs_accordion.get_config()
                )
            )
        else:
            raise ValueError(
                f"The `additional_inputs_accordion` parameter must be a string or gr.Accordion, not {type(additional_inputs_accordion)}"
            )

        with self:
            if title:
                Markdown(
                    f"<h1 style='text-align: center; margin-bottom: 1rem'>{self.title}</h1>"
                )
            if description:
                Markdown(description)

            self.chatbot = chatbot.render()

            self.buttons = [retry_btn, undo_btn, clear_btn]

            with Group():
                with Row():
                    if textbox:
                        if self.multimodal:
                            submit_btn = None
                        else:
                            textbox.container = False
                        textbox.show_label = False
                        textbox_ = textbox.render()
                        if not isinstance(textbox_, (Textbox, MultimodalTextbox)):
                            raise TypeError(
                                f"Expected a gr.Textbox or gr.MultimodalTextbox component, but got {type(textbox_)}"
                            )
                        self.textbox = textbox_
                    elif self.multimodal:
                        submit_btn = None
                        self.textbox = MultimodalTextbox(
                            show_label=False,
                            label="Message",
                            placeholder="Type a message...",
                            scale=7,
                            autofocus=autofocus,
                        )
                    else:
                        self.textbox = Textbox(
                            container=False,
                            show_label=False,
                            label="Message",
                            placeholder="Type a message...",
                            scale=7,
                            autofocus=autofocus,
                        )
                    if submit_btn is not None and not multimodal:
                        if isinstance(submit_btn, Button):
                            submit_btn.render()
                        elif isinstance(submit_btn, str):
                            submit_btn = Button(
                                submit_btn,
                                variant="primary",
                                scale=1,
                                min_width=150,
                            )
                        else:
                            raise ValueError(
                                f"The submit_btn parameter must be a gr.Button, string, or None, not {type(submit_btn)}"
                            )
                    if stop_btn is not None:
                        if isinstance(stop_btn, Button):
                            stop_btn.visible = False
                            stop_btn.render()
                        elif isinstance(stop_btn, str):
                            stop_btn = Button(
                                stop_btn,
                                variant="stop",
                                visible=False,
                                scale=1,
                                min_width=150,
                            )
                        else:
                            raise ValueError(
                                f"The stop_btn parameter must be a gr.Button, string, or None, not {type(stop_btn)}"
                            )
                    self.buttons.extend([submit_btn, stop_btn])  # type: ignore

                self.fake_api_btn = Button("Fake API", visible=False)
                self.fake_response_textbox = Textbox(label="Response", visible=False)
                (
                    self.retry_btn,
                    self.undo_btn,
                    self.clear_btn,
                    self.submit_btn,
                    self.stop_btn,
                ) = self.buttons

            any_unrendered_inputs = any(
                not inp.is_rendered for inp in self.additional_inputs
            )
            if self.additional_inputs and any_unrendered_inputs:
                with Accordion(**self.additional_inputs_accordion_params):  # type: ignore
                    for input_component in self.additional_inputs:
                        if not input_component.is_rendered:
                            input_component.render()

            self.saved_input = State()
            self.chatbot_state = (
                State(self.chatbot.value) if self.chatbot.value else State([])
            )

            self._setup_events()
            self._setup_api()

        if examples:
            examples.click(lambda x: x[0], inputs=[examples], outputs=self.textbox, show_progress=False, queue=False)

    def _setup_events(self) -> None:
        submit_fn = self._stream_fn if self.is_generator else self._submit_fn
        submit_triggers = (
            [self.textbox.submit, self.submit_btn.click]
            if self.submit_btn
            else [self.textbox.submit]
        )
        submit_event = (
            on(
                submit_triggers,
                self._clear_and_save_textbox,
                [self.textbox],
                [self.textbox, self.saved_input],
                show_api=False,
                queue=False,
            )
            .then(
                self.pre_fn,
                **self.pre_fn_kwargs,
                show_api=False,
                queue=False,
            )
            .then(
                self._display_input,
                [self.saved_input, self.chatbot_state],
                [self.chatbot, self.chatbot_state],
                show_api=False,
                queue=False,
            )
            .then(
                submit_fn,
                [self.saved_input, self.chatbot_state] + self.additional_inputs,
                [self.chatbot, self.chatbot_state, self.interrupter],
                show_api=False,
                concurrency_limit=cast(
                    Union[int, Literal["default"], None], self.concurrency_limit
                ),
            ).then(
                self.post_fn,
                **self.post_fn_kwargs,
                show_api=False,
                concurrency_limit=cast(
                    Union[int, Literal["default"], None], self.concurrency_limit
                ),
            )
        )
        self._setup_stop_events(submit_triggers, submit_event)

        if self.retry_btn:
            retry_event = (
                self.retry_btn.click(
                    self._delete_prev_fn,
                    [self.saved_input, self.chatbot_state],
                    [self.chatbot, self.saved_input, self.chatbot_state],
                    show_api=False,
                    queue=False,
                )
                .then(
                    self.pre_fn,
                    **self.pre_fn_kwargs,
                    show_api=False,
                    queue=False,
                )
                .then(
                    self._display_input,
                    [self.saved_input, self.chatbot_state],
                    [self.chatbot, self.chatbot_state],
                    show_api=False,
                    queue=False,
                )
                .then(
                    submit_fn,
                    [self.saved_input, self.chatbot_state] + self.additional_inputs,
                    [self.chatbot, self.chatbot_state],
                    show_api=False,
                    concurrency_limit=cast(
                        Union[int, Literal["default"], None], self.concurrency_limit
                    ),
                ).then(
                self.post_fn,
                **self.post_fn_kwargs,
                show_api=False,
                concurrency_limit=cast(
                    Union[int, Literal["default"], None], self.concurrency_limit
                ),
            )
            )
            self._setup_stop_events([self.retry_btn.click], retry_event)

        if self.undo_btn:
            self.undo_btn.click(
                self._delete_prev_fn,
                [self.saved_input, self.chatbot_state],
                [self.chatbot, self.saved_input, self.chatbot_state],
                show_api=False,
                queue=False,
            ).then(
                self.pre_fn,
                **self.pre_fn_kwargs,
                show_api=False,
                queue=False,
            ).then(
                async_lambda(lambda x: x),
                [self.saved_input],
                [self.textbox],
                show_api=False,
                queue=False,
            ).then(
                self.post_fn,
                **self.post_fn_kwargs,
                show_api=False,
                concurrency_limit=cast(
                    Union[int, Literal["default"], None], self.concurrency_limit
                ),
            )

        if self.clear_btn:
            self.clear_btn.click(
                async_lambda(lambda: ([], [], None)),
                None,
                [self.chatbot, self.chatbot_state, self.saved_input],
                queue=False,
                show_api=False,
            ).then(
                self.pre_fn,
                **self.pre_fn_kwargs,
                show_api=False,
                queue=False,
            ).then(
                self.post_fn,
                **self.post_fn_kwargs,
                show_api=False,
                concurrency_limit=cast(
                    Union[int, Literal["default"], None], self.concurrency_limit
                ),
            )

    def _setup_stop_events(
        self, event_triggers: list[Callable], event_to_cancel: Dependency
    ) -> None:
        def perform_interrupt(ipc):
            if ipc is not None:
                ipc()
            return

        if self.stop_btn and self.is_generator:
            if self.submit_btn:
                for event_trigger in event_triggers:
                    event_trigger(
                        async_lambda(
                            lambda: (
                                Button(visible=False),
                                Button(visible=True),
                            )
                        ),
                        None,
                        [self.submit_btn, self.stop_btn],
                        show_api=False,
                        queue=False,
                    )
                event_to_cancel.then(
                    async_lambda(lambda: (Button(visible=True), Button(visible=False))),
                    None,
                    [self.submit_btn, self.stop_btn],
                    show_api=False,
                    queue=False,
                )
            else:
                for event_trigger in event_triggers:
                    event_trigger(
                        async_lambda(lambda: Button(visible=True)),
                        None,
                        [self.stop_btn],
                        show_api=False,
                        queue=False,
                    )
                event_to_cancel.then(
                    async_lambda(lambda: Button(visible=False)),
                    None,
                    [self.stop_btn],
                    show_api=False,
                    queue=False,
                )
            self.stop_btn.click(
                fn=perform_interrupt,
                inputs=[self.interrupter],
                cancels=event_to_cancel,
                show_api=False,
            )

    def _setup_api(self) -> None:
        api_fn = self._api_stream_fn if self.is_generator else self._api_submit_fn

        self.fake_api_btn.click(
            api_fn,
            [self.textbox, self.chatbot_state] + self.additional_inputs,
            [self.textbox, self.chatbot_state],
            api_name="chat",
            concurrency_limit=cast(
                Union[int, Literal["default"], None], self.concurrency_limit
            ),
        )

    def _clear_and_save_textbox(self, message: str) -> tuple[str | dict, str]:
        if self.multimodal:
            return {"text": "", "files": []}, message
        else:
            return "", message

    def _append_multimodal_history(
        self,
        message: dict[str, list],
        response: str | None,
        history: list[list[str | tuple | None]],
    ):
        for x in message["files"]:
            history.append([(x,), None])
        if message["text"] is None or not isinstance(message["text"], str):
            return
        elif message["text"] == "" and message["files"] != []:
            history.append([None, response])
        else:
            history.append([message["text"], response])

    async def _display_input(
        self, message: str | dict[str, list], history: list[list[str | tuple | None]]
    ) -> tuple[list[list[str | tuple | None]], list[list[str | tuple | None]]]:
        if self.multimodal and isinstance(message, dict):
            self._append_multimodal_history(message, None, history)
        elif isinstance(message, str):
            history.append([message, None])
        return history, history

    async def _submit_fn(
        self,
        message: str | dict[str, list],
        history_with_input: list[list[str | tuple | None]],
        request: Request,
        *args,
    ) -> tuple[list[list[str | tuple | None]], list[list[str | tuple | None]]]:
        if self.multimodal and isinstance(message, dict):
            remove_input = (
                len(message["files"]) + 1
                if message["text"] is not None
                else len(message["files"])
            )
            history = history_with_input[:-remove_input]
        else:
            history = history_with_input[:-1]
        inputs, _, _ = special_args(
            self.fn, inputs=[message, history, *args], request=request
        )

        if self.is_async:
            response = await self.fn(*inputs)
        else:
            response = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )

        if self.multimodal and isinstance(message, dict):
            self._append_multimodal_history(message, response, history)
        elif isinstance(message, str):
            history.append([message, response])
        return history, history

    async def _stream_fn(
        self,
        message: str | dict[str, list],
        history_with_input: list[list[str | tuple | None]],
        request: Request,
        *args,
    ) -> AsyncGenerator:
        if self.multimodal and isinstance(message, dict):
            remove_input = (
                len(message["files"]) + 1
                if message["text"] is not None
                else len(message["files"])
            )
            history = history_with_input[:-remove_input]
        else:
            history = history_with_input[:-1]
        inputs, _, _ = special_args(
            self.fn, inputs=[message, history, *args], request=request
        )

        if self.is_async:
            generator = self.fn(*inputs)
        else:
            generator = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )
            generator = SyncToAsyncIterator(generator, self.limiter)
        try:
            first_response, first_interrupter = await async_iteration(generator)
            if self.multimodal and isinstance(message, dict):
                for x in message["files"]:
                    history.append([(x,), None])
                update = history + [[message["text"], first_response]]
                yield update, update
            else:
                update = history + [[message, first_response]]
                yield update, update, first_interrupter
        except StopIteration:
            if self.multimodal and isinstance(message, dict):
                self._append_multimodal_history(message, None, history)
                yield history, history
            else:
                update = history + [[message, None]]
                yield update, update, first_interrupter
        async for response, interrupter in generator:
            if self.multimodal and isinstance(message, dict):
                update = history + [[message["text"], response]]
                yield update, update
            else:
                update = history + [[message, response]]
                yield update, update, interrupter

    async def _api_submit_fn(
        self, message: str, history: list[list[str | None]], request: Request, *args
    ) -> tuple[str, list[list[str | None]]]:
        inputs, _, _ = special_args(
            self.fn, inputs=[message, history, *args], request=request
        )

        if self.is_async:
            response = await self.fn(*inputs)
        else:
            response = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )
        history.append([message, response])
        return response, history

    async def _api_stream_fn(
        self, message: str, history: list[list[str | None]], request: Request, *args
    ) -> AsyncGenerator:
        inputs, _, _ = special_args(
            self.fn, inputs=[message, history, *args], request=request
        )

        if self.is_async:
            generator = self.fn(*inputs)
        else:
            generator = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )
            generator = SyncToAsyncIterator(generator, self.limiter)
        try:
            first_response = await async_iteration(generator)
            yield first_response, history + [[message, first_response]]
        except StopIteration:
            yield None, history + [[message, None]]
        async for response in generator:
            yield response, history + [[message, response]]

    async def _delete_prev_fn(
        self,
        message: str | dict[str, list],
        history: list[list[str | tuple | None]],
    ) -> tuple[
        list[list[str | tuple | None]],
        str | dict[str, list],
        list[list[str | tuple | None]],
    ]:
        if self.multimodal and isinstance(message, dict):
            remove_input = (
                len(message["files"]) + 1
                if message["text"] is not None
                else len(message["files"])
            )
            history = history[:-remove_input]
        else:
            while history:
                deleted_a, deleted_b = history[-1]
                history = history[:-1]
                if isinstance(deleted_a, str) and isinstance(deleted_b, str):
                    break
        return history, message or "", history
