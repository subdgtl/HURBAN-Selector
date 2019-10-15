use std::thread;

use crossbeam_channel as channel;

use crate::interpreter::ast::{Prog, Stmt};
use crate::interpreter::{InterpretError, Interpreter, Value};
use crate::interpreter_funcs;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(u64);

/// A possible error when polling for an interpreter response.
#[derive(Debug)]
pub enum PollResponseError {
    /// The interpreter didn't send a response yet. The caller should
    /// retry later.
    Pending,
}

/// An interpreter request. Variants correspond to various methods on
/// the interpreter.
#[derive(Debug)]
pub enum InterpreterRequest {
    SetProg(Prog),
    #[allow(dead_code)]
    ClearProg,
    #[allow(dead_code)]
    PushProgStmt(Stmt),
    #[allow(dead_code)]
    PopProgStmt,
    #[allow(dead_code)]
    SetProgStmtAt(usize, Stmt),
    Interpret,
    #[allow(dead_code)]
    InterpretUpUntil(usize),
}

/// An interpreter response.
#[derive(Debug)]
pub enum InterpreterResponse {
    /// Interpreter successfully completed request.
    Completed,

    /// Interpreter successfully completed request and responded with
    /// a result.
    CompletedWithResult(Result<Value, InterpretError>),
}

enum Request {
    Command {
        request_id: RequestId,
        data: InterpreterRequest,
    },
    Shutdown,
}

struct Response {
    request_id: RequestId,
    data: InterpreterResponse,
}

/// An asynchronous server for an interpreter instance running in a
/// separate thread.
///
/// Allows for sending requests to the interpreter and waiting for
/// related responses.
pub struct InterpreterServer {
    next_request_id: u64,
    thread: Option<thread::JoinHandle<()>>,
    request_sender: channel::Sender<Request>,
    response_receiver: channel::Receiver<Response>,
}

impl InterpreterServer {
    pub fn new() -> Self {
        let (request_sender, request_receiver) = channel::unbounded();
        let (response_sender, response_receiver) = channel::unbounded();

        let thread = thread::spawn(move || {
            log::info!("Interpreter server starting up");

            let mut interpreter = Interpreter::new(interpreter_funcs::global_definitions());

            loop {
                let request: Request = request_receiver
                    .recv()
                    .expect("Interpreter server failed to receive request");

                let (request_id, data) = match request {
                    Request::Command { request_id, data } => (request_id, data),
                    Request::Shutdown => break,
                };

                // FIXME: handle potential interpreter panic?

                let response = match data {
                    InterpreterRequest::SetProg(prog) => {
                        log::info!(
                            "Interpreter server received request 'SetProg' with {}",
                            prog,
                        );
                        interpreter.set_prog(prog);
                        Response {
                            request_id,
                            data: InterpreterResponse::Completed,
                        }
                    }
                    InterpreterRequest::ClearProg => {
                        log::info!("Interpreter server received request 'ClearProg'");
                        interpreter.clear_prog();
                        Response {
                            request_id,
                            data: InterpreterResponse::Completed,
                        }
                    }
                    InterpreterRequest::PushProgStmt(stmt) => {
                        log::info!(
                            "Interpreter server received request 'PushProgStmt' with {}",
                            stmt,
                        );
                        interpreter.push_prog_stmt(stmt);
                        Response {
                            request_id,
                            data: InterpreterResponse::Completed,
                        }
                    }
                    InterpreterRequest::PopProgStmt => {
                        log::info!("Interpreter server received request 'PopProgStmt'");
                        interpreter.pop_prog_stmt();
                        Response {
                            request_id,
                            data: InterpreterResponse::Completed,
                        }
                    }
                    InterpreterRequest::SetProgStmtAt(index, stmt) => {
                        log::info!(
                            "Interpreter server received request 'SetProgStmtAt({})' with {}",
                            index,
                            stmt,
                        );
                        interpreter.set_prog_stmt_at(index, stmt);
                        Response {
                            request_id,
                            data: InterpreterResponse::Completed,
                        }
                    }
                    InterpreterRequest::Interpret => {
                        log::info!("Interpreter server received request 'Interpret'");
                        let result = interpreter.interpret();
                        Response {
                            request_id,
                            data: InterpreterResponse::CompletedWithResult(result),
                        }
                    }
                    InterpreterRequest::InterpretUpUntil(index) => {
                        log::info!(
                            "Interpreter server received request 'InterpretUpUntil({})'",
                            index,
                        );
                        let result = interpreter.interpret_up_until(index);
                        Response {
                            request_id,
                            data: InterpreterResponse::CompletedWithResult(result),
                        }
                    }
                };

                response_sender
                    .send(response)
                    .expect("Interpreter server failed to send response");
            }

            log::info!("Interpreter server shutting down");
        });

        Self {
            next_request_id: 0,
            thread: Some(thread),
            request_sender,
            response_receiver,
        }
    }

    /// Submit a new request for the interpreter to work on.
    ///
    /// The corresponding response can be paired with based on the
    /// provided request id.
    pub fn submit_request(&mut self, request: InterpreterRequest) -> RequestId {
        let request_id = RequestId(self.next_request_id);
        self.next_request_id += 1;

        self.request_sender
            .send(Request::Command {
                request_id,
                data: request,
            })
            .expect("Interpreter client failed to send request");

        request_id
    }

    /// Poll the server for a possible response.
    ///
    /// In case of no response, [`PollResponseError::Pending`] is returned.
    ///
    /// [`PollResponseError::Pending`]: enum.PollresponseError.html#variant.Pending
    pub fn poll_response(&self) -> Result<(RequestId, InterpreterResponse), PollResponseError> {
        if self.response_receiver.is_empty() {
            Err(PollResponseError::Pending)
        } else {
            let response = self
                .response_receiver
                .recv()
                .expect("Interpreter client failed to receive response");

            Ok((response.request_id, response.data))
        }
    }
}

impl Drop for InterpreterServer {
    fn drop(&mut self) {
        self.request_sender
            .send(Request::Shutdown)
            .expect("Interpreter client failed to send shutdown request");

        if let Some(thread) = self.thread.take() {
            log::info!("Waiting for interpreter server to shut down");
            thread
                .join()
                .expect("Interpreter thread panicked before joining");
        }
    }
}
