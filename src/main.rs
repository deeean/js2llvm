use std::collections::HashMap;
use std::path::Path;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::{Linkage, Module};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel};
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::types::{BasicMetadataTypeEnum, StructType};
use inkwell::values::{BasicValue, FunctionValue, GlobalValue, PointerValue};

pub type Program = Vec<Statement>;

pub enum Value {
    Number(f64),
    Boolean(bool),
    String(String),
    // Object(HashMap<String, Value>),
    // Array(Vec<Value>),
    Null,
    Undefined,
}

pub enum Statement {
    Assignment(String, Expression),
    VariableDeclaration(String, Expression),
    FunctionDeclaration(String, Vec<String>, Program),
    Expression(Expression),
    Return(Expression),
    While(Expression, Program),
}

pub enum BinaryOperator {
    Add,
    LessThan,
}

pub enum Expression {
    Literal(Value),
    Identifier(String),
    Binary(BinaryOperator, Box<Expression>, Box<Expression>),
    Call(String, Vec<Expression>),
}

// ===

#[repr(u8)]
pub enum Type {
    Number = 0,
    Boolean = 1,
    String = 2,
    Object = 3,
    Array = 4,
    Null = 5,
    Undefined = 6,
}

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    variables: HashMap<String, PointerValue<'ctx>>,
    execution_engine: ExecutionEngine<'ctx>,
    string_constants: HashMap<String, GlobalValue<'ctx>>,
    js_value_type: StructType<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let module = context.create_module("main");
        let builder = context.create_builder();

        let execution_engine = module.create_jit_execution_engine(OptimizationLevel::Aggressive).unwrap();

        let i8_type = context.i8_type();
        let f64_type = context.f64_type();
        let bool_type = context.bool_type();
        let ptr_type = context.ptr_type(AddressSpace::default());
        let null_bool_type = context.bool_type();
        let undefined_bool_type = context.bool_type();
        let js_value_type = context.opaque_struct_type("js_value");

        js_value_type.set_body(&[
            i8_type.into(),
            f64_type.into(),
            bool_type.into(),
            ptr_type.into(),
            null_bool_type.into(),
            undefined_bool_type.into(),
        ], false);

        Self {
            context,
            module,
            builder,
            variables: HashMap::new(),
            execution_engine,
            string_constants: HashMap::new(),
            js_value_type
        }
    }

    fn create_entry_block_alloca(&self, r#type: StructType<'ctx>, name: &str) -> PointerValue<'ctx> {
        let current_function = self.get_current_function();
        let entry_block = current_function.get_first_basic_block().unwrap();
        let builder = self.context.create_builder();

        if let Some(first_instruction) = entry_block.get_first_instruction() {
            builder.position_before(&first_instruction);
        } else {
            builder.position_at_end(entry_block);
        }

        builder.build_alloca(r#type, name).unwrap()
    }

    fn declare_printf_func(&mut self) {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let printf_type = self.context.i32_type().fn_type(&[ptr_type.into()], true);
        self.module.add_function("printf", printf_type, Some(Linkage::External));
    }

    fn get_function(&self, name: &str) -> Option<FunctionValue<'ctx>> {
        self.module.get_function(name)
    }

    fn build_global_string_ptr(&mut self, name: &str, value: &str) -> GlobalValue<'ctx> {
        if let Some(string_ptr) = self.string_constants.get(name) {
            return *string_ptr;
        }

        let string_ptr = self.builder.build_global_string_ptr(name, value).unwrap();
        self.string_constants.insert(name.to_string(), string_ptr);
        string_ptr
    }

    fn generate_println(&mut self, args: &Vec<Expression>) -> PointerValue<'ctx> {
        let printf_func = self.get_function("printf").unwrap();

        for arg in args {
            let js_value = match self.generate_expression(arg) {
                Some(value) => value,
                None => continue,
            };

            let js_value_type = self.js_value_type;

            let type_ptr = self.builder
                .build_struct_gep(js_value_type, js_value, 0, "type_ptr")
                .unwrap();
            let type_tag = self.builder
                .build_load(self.context.i8_type(), type_ptr, "type_tag")
                .unwrap();

            let current_function = self.get_current_function();
            let number_block = self.context.append_basic_block(current_function, "number_block");
            let boolean_block = self.context.append_basic_block(current_function, "boolean_block");
            let string_block = self.context.append_basic_block(current_function, "string_block");
            let null_block = self.context.append_basic_block(current_function, "null_block");
            let undefined_block = self.context.append_basic_block(current_function, "undefined_block");
            let after_block = self.context.append_basic_block(current_function, "after_block");

            self.builder.build_switch(
                type_tag.into_int_value(),
                undefined_block,
                &[
                    (
                        self.context.i8_type().const_int(Type::Number as u64, false),
                        number_block,
                    ),
                    (
                        self.context.i8_type().const_int(Type::Boolean as u64, false),
                        boolean_block,
                    ),
                    (
                        self.context.i8_type().const_int(Type::String as u64, false),
                        string_block,
                    ),
                    (
                        self.context.i8_type().const_int(Type::Null as u64, false),
                        null_block,
                    ),
                    (
                        self.context.i8_type().const_int(Type::Undefined as u64, false),
                        undefined_block,
                    ),
                ],
            ).unwrap();

            self.builder.position_at_end(number_block);
            let value_ptr = self.builder
                .build_struct_gep(js_value_type, js_value, 1, "number_value_ptr")
                .unwrap();
            let number_value = self.builder
                .build_load(self.context.f64_type(), value_ptr, "number_value")
                .unwrap();
            let format_str = self.build_global_string_ptr("%f ", "number_format");
            let args = [format_str.as_pointer_value().into(), number_value.into()];
            self.builder.build_call(printf_func, &args, "printf_call").unwrap();
            self.builder.build_unconditional_branch(after_block).unwrap();

            // Boolean case
            self.builder.position_at_end(boolean_block);
            let value_ptr = self.builder
                .build_struct_gep(js_value_type, js_value, 2, "boolean_value_ptr")
                .unwrap();
            let bool_value = self.builder
                .build_load(self.context.bool_type(), value_ptr, "bool_value")
                .unwrap();
            let true_str = self.build_global_string_ptr("true ", "true_str");
            let false_str = self.build_global_string_ptr("false ", "false_str");
            let selected_str = self.builder.build_select(
                bool_value.into_int_value(),
                true_str.as_pointer_value(),
                false_str.as_pointer_value(),
                "selected_str",
            ).unwrap();
            let format_str = self.build_global_string_ptr("%s", "bool_format");
            let args = [format_str.as_pointer_value().into(), selected_str.into()];
            self.builder.build_call(printf_func, &args, "printf_call").unwrap();
            self.builder.build_unconditional_branch(after_block).unwrap();

            // String case
            self.builder.position_at_end(string_block);
            let value_ptr = self.builder
                .build_struct_gep(js_value_type, js_value, 3, "string_value_ptr")
                .unwrap();
            let string_ptr = self.builder
                .build_load(self.context.ptr_type(AddressSpace::default()), value_ptr, "string_ptr")
                .unwrap();
            let format_str = self.build_global_string_ptr("%s ", "string_format");
            let args = [format_str.as_pointer_value().into(), string_ptr.into()];
            self.builder.build_call(printf_func, &args, "printf_call").unwrap();
            self.builder.build_unconditional_branch(after_block).unwrap();

            // Null case
            self.builder.position_at_end(null_block);
            let format_str = self.build_global_string_ptr("null ", "null_format");
            let args = [format_str.as_pointer_value().into()];
            self.builder.build_call(printf_func, &args, "printf_call").unwrap();
            self.builder.build_unconditional_branch(after_block).unwrap();

            // Undefined case
            self.builder.position_at_end(undefined_block);
            let format_str = self.build_global_string_ptr("undefined ", "undefined_format");
            let args = [format_str.as_pointer_value().into()];
            self.builder.build_call(printf_func, &args, "printf_call").unwrap();
            self.builder.build_unconditional_branch(after_block).unwrap();

            self.builder.position_at_end(after_block);
        }

        let newline_format = self.build_global_string_ptr("\n", "newline_format");
        let args = [newline_format.as_pointer_value().into()];
        self.builder.build_call(printf_func, &args, "newline_call").unwrap();

        self.generate_js_value(&Value::Undefined)
    }

    fn get_current_function(&self) -> FunctionValue<'ctx> {
        self.builder.get_insert_block().unwrap().get_parent().unwrap().into()
    }

    fn generate_js_value(&mut self, value: &Value) -> PointerValue<'ctx> {
        let js_value_type = self.js_value_type;
        let js_value = self.create_entry_block_alloca(js_value_type, "js_value");

        match value {
            Value::Number(number) => {
                let type_tag = self.context.i8_type().const_int(Type::Number as u64, false);
                let number_value = self.context.f64_type().const_float(*number);

                let type_ptr = self.builder.build_struct_gep(js_value_type, js_value, 0, "type_ptr").unwrap();
                let value_ptr = self.builder.build_struct_gep(js_value_type, js_value, 1, "value_ptr").unwrap();

                self.builder.build_store(type_ptr, type_tag).unwrap();
                self.builder.build_store(value_ptr, number_value).unwrap();
            }
            Value::Boolean(boolean) => {
                let type_tag = self.context.i8_type().const_int(Type::Boolean as u64, false);
                let boolean_value = self.context.bool_type().const_int(*boolean as u64, false);

                let type_ptr = self.builder.build_struct_gep(js_value_type, js_value, 0, "type_ptr").unwrap();
                let value_ptr = self.builder.build_struct_gep(js_value_type, js_value, 2, "value_ptr").unwrap();

                self.builder.build_store(type_ptr, type_tag).unwrap();
                self.builder.build_store(value_ptr, boolean_value).unwrap();
            }
            Value::String(string) => {
                let type_tag = self.context.i8_type().const_int(Type::String as u64, false);
                let string_value = self.build_global_string_ptr(string, "string_value");

                let type_ptr = self.builder.build_struct_gep(js_value_type, js_value, 0, "type_ptr").unwrap();
                let value_ptr = self.builder.build_struct_gep(js_value_type, js_value, 3, "value_ptr").unwrap();

                self.builder.build_store(type_ptr, type_tag).unwrap();
                self.builder.build_store(value_ptr, string_value.as_pointer_value()).unwrap();
            }
            Value::Null => {
                let type_tag = self.context.i8_type().const_int(Type::Null as u64, false);
                let null_value = self.context.bool_type().const_int(1, false);

                let type_ptr = self.builder.build_struct_gep(js_value_type, js_value, 0, "type_ptr").unwrap();
                let value_ptr = self.builder.build_struct_gep(js_value_type, js_value, 4, "value_ptr").unwrap();

                self.builder.build_store(type_ptr, type_tag).unwrap();
                self.builder.build_store(value_ptr, null_value).unwrap();
            }
            Value::Undefined => {
                let type_tag = self.context.i8_type().const_int(Type::Undefined as u64, false);
                let undefined_value = self.context.bool_type().const_int(1, false);

                let type_ptr = self.builder.build_struct_gep(js_value_type, js_value, 0, "type_ptr").unwrap();
                let value_ptr = self.builder.build_struct_gep(js_value_type, js_value, 5, "value_ptr").unwrap();

                self.builder.build_store(type_ptr, type_tag).unwrap();
                self.builder.build_store(value_ptr, undefined_value).unwrap();
            }
        }

        js_value
    }

    fn get_js_value_type(&self, ptr: PointerValue<'ctx>) -> Type {
        let type_ptr = self.builder.build_struct_gep(self.js_value_type, ptr, 0, "type_ptr").unwrap();
        let type_val = self.builder.build_load(self.context.i8_type(), type_ptr, "type").unwrap().into_int_value();
        let num_type = type_val.get_type().const_int(Type::Number as u64, false);
        let is_num = self.builder.build_int_compare(IntPredicate::EQ, type_val, num_type, "is_num").unwrap();
        let current_function = self.get_current_function();
        let then_block = self.context.append_basic_block(current_function, "then");
        let else_block = self.context.append_basic_block(current_function, "else");
        let merge_block = self.context.append_basic_block(current_function, "merge");
        let result = self.context.i8_type().const_int(0, false);

        self.builder.build_conditional_branch(is_num, then_block, else_block).unwrap();

        self.builder.position_at_end(then_block);
        self.builder.build_unconditional_branch(merge_block).unwrap();

        self.builder.position_at_end(else_block);
        self.builder.build_unconditional_branch(merge_block).unwrap();

        self.builder.position_at_end(merge_block);
        Type::Number
    }

    fn generate_expression(&mut self, expression: &Expression) -> Option<PointerValue<'ctx>> {
        match expression {
            Expression::Literal(value) => {
                Some(self.generate_js_value(value))
            }
            Expression::Identifier(name) => {
                match self.variables.get(name) {
                    Some(value) => Some(*value),
                    None => panic!("Variable {} not found", name),
                }
            }
            Expression::Binary(op, left, right) => {
                let left_val = self.generate_expression(left)?;
                let right_val = self.generate_expression(right)?;

                let left_type = self.get_js_value_type(left_val);
                let right_type = self.get_js_value_type(right_val);

                match (left_type, right_type) {
                    (Type::Number, Type::Number) => {
                        let l = self.load_number(left_val);
                        let r = self.load_number(right_val);
                        let result = match op {
                            BinaryOperator::Add => self.builder.build_float_add(l, r, "add").unwrap(),
                            BinaryOperator::LessThan => {
                                let cmp = self.builder.build_float_compare(FloatPredicate::ULT, l, r, "cmp").unwrap();
                                // self.builder.build_zext(cmp, self.context.i8_type(), "bool_ext").unwrap().into_float_value()
                                self.builder.build_unsigned_int_to_float(cmp, self.context.f64_type(), "bool_ext").unwrap()
                            }
                        };
                        let result_val = self.create_entry_block_alloca(self.js_value_type, "result");
                        let type_ptr = self.builder.build_struct_gep(self.js_value_type, result_val, 0, "type_ptr").unwrap();
                        self.builder.build_store(type_ptr, self.context.i8_type().const_int(Type::Number as u64, false)).unwrap();
                        let value_ptr = self.builder.build_struct_gep(self.js_value_type, result_val, 1, "value_ptr").unwrap();
                        self.builder.build_store(value_ptr, result).unwrap();
                        Some(result_val)
                    }
                    _ => unimplemented!(),
                }
            }

            Expression::Call(name, args) => {
                if name == "println" {
                    Some(self.generate_println(args))
                } else {
                    let function = self.get_function(name)
                        .unwrap_or_else(|| panic!("Function {} not found", name));
                    let js_value_type = self.js_value_type;

                    let mut llvm_args = Vec::new();
                    for arg in args {
                        let arg_ptr = self.generate_expression(arg)
                            .expect("Failed to generate argument");
                        let loaded_arg = self.builder.build_load(js_value_type, arg_ptr, "arg_load").unwrap();
                        llvm_args.push(loaded_arg.into());
                    }

                    let call_inst = self.builder.build_call(function, &llvm_args, "call").unwrap();
                    let result_ptr = self.create_entry_block_alloca(js_value_type, "result");
                    let returned_val = call_inst
                        .try_as_basic_value()
                        .left()
                        .expect("Call did not return a value");

                    self.builder.build_store(result_ptr, returned_val).unwrap();

                    Some(result_ptr)
                }
            }
            _ => unimplemented!(),
        }
    }

    fn load_number(&self, ptr: PointerValue<'ctx>) -> inkwell::values::FloatValue<'ctx> {
        let value_ptr = self.builder.build_struct_gep(self.js_value_type, ptr, 1, "num_ptr").unwrap();
        self.builder.build_load(self.context.f64_type(), value_ptr, "num").unwrap().into_float_value()
    }

    fn generate_statement(&mut self, statement: &Statement) {
        match statement {
            Statement::While(cond, body) => {
                let current_function = self.get_current_function();
                let cond_block = self.context.append_basic_block(current_function, "while_cond");
                let body_block = self.context.append_basic_block(current_function, "while_body");
                let exit_block = self.context.append_basic_block(current_function, "while_exit");

                self.builder.build_unconditional_branch(cond_block).unwrap();
                self.builder.position_at_end(cond_block);

                let cond_val = self.generate_expression(cond).unwrap();
                let cond_num = self.load_number(cond_val);
                let zero = self.context.f64_type().const_float(0.0);
                let cond = self.builder.build_float_compare(FloatPredicate::ONE, cond_num, zero, "loop_cond").unwrap();

                self.builder.build_conditional_branch(cond, body_block, exit_block).unwrap();

                self.builder.position_at_end(body_block);
                for s in body {
                    self.generate_statement(s);
                }
                self.builder.build_unconditional_branch(cond_block).unwrap();

                self.builder.position_at_end(exit_block);
            }
            Statement::Assignment(name, expression) => {
                let new_value_ptr = self.generate_expression(expression).unwrap();
                let js_value_type = self.js_value_type;

                let new_value = self.builder.build_load(js_value_type, new_value_ptr, "new_value").unwrap();

                if let Some(old_value_ptr) = self.variables.get(name) {
                    self.builder.build_store(*old_value_ptr, new_value).unwrap();
                } else {
                    panic!("Variable {} not found", name);
                }
            }
            Statement::FunctionDeclaration(name, params, body) => {
                let js_value_type = self.js_value_type;

                let param_types: Vec<BasicMetadataTypeEnum> = params.iter().map(|_| js_value_type.into()).collect();

                let function_type = js_value_type.fn_type(&param_types, false);
                let function = self.module.add_function(name, function_type, None);

                let basic_block = self.context.append_basic_block(function, "entry");
                self.builder.position_at_end(basic_block);

                let old_variables = std::mem::take(&mut self.variables);

                for (i, param) in params.iter().enumerate() {
                    let js_value_type = self.js_value_type;
                    let param_value = function.get_nth_param(i as u32).unwrap();

                    param_value.set_name(param);

                    let param_ptr = self.create_entry_block_alloca(js_value_type, param);
                    self.builder.build_store(param_ptr, param_value).unwrap();
                    self.variables.insert(param.to_string(), param_ptr);
                }

                for statement in body {
                    self.generate_statement(statement);
                }

                if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
                    let undefined_ptr = self.generate_js_value(&Value::Undefined);
                    let loaded_value = self.builder.build_load(js_value_type, undefined_ptr, "return_value").unwrap();
                    self.builder.build_return(Some(&loaded_value)).unwrap();
                }

                self.variables = old_variables;
            }
            Statement::VariableDeclaration(name, expression) => {
                let value = self.generate_expression(expression).unwrap();
                self.variables.insert(name.clone(), value);
            }
            Statement::Expression(expression) => {
                self.generate_expression(expression);
            }
            Statement::Return(expression) => {
                let js_value_type = self.js_value_type;
                let value = self.generate_expression(expression).unwrap();
                let loaded_value = self.builder.build_load(js_value_type, value, "return_value").unwrap();
                let basic_value = loaded_value.as_basic_value_enum();
                self.builder.build_return(Some(&basic_value)).unwrap();
            }
        }
    }

    fn init_main_function(&self) -> FunctionValue<'ctx> {
        let i32_type = self.context.i32_type();
        let main_type = i32_type.fn_type(&[], false);
        let main_func = self.module.add_function("main", main_type, None);
        let basic_block = self.context.append_basic_block(main_func, "entry");
        self.builder.position_at_end(basic_block);

        main_func
    }

    pub fn generate(&mut self, program: &Program) {
        self.declare_printf_func();

        self.init_main_function();

        for statement in program {
            self.generate_statement(statement);
        }

        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
            let i32_type = self.context.i32_type();
            let zero = i32_type.const_zero();
            self.builder.build_return(Some(&zero)).unwrap();
        }

        self.module.print_to_file(Path::new("output.ll")).unwrap();
    }
}

fn main() {
    let program: Program = vec![
        Statement::VariableDeclaration("i".to_string(), Expression::Literal(Value::Number(0.0))),
        Statement::VariableDeclaration("sum".to_string(), Expression::Literal(Value::Number(0.0))),
        Statement::While(
            Expression::Binary(
                BinaryOperator::LessThan,
                Box::new(Expression::Identifier("i".to_string())),
                Box::new(Expression::Literal(Value::Number(100_000_000.0))),
            ),
            vec![
                Statement::Assignment(
                    "i".to_string(),
                    Expression::Binary(
                        BinaryOperator::Add,
                        Box::new(Expression::Identifier("i".to_string())),
                        Box::new(Expression::Literal(Value::Number(1.0))),
                    )
                ),
                Statement::Assignment(
                    "sum".to_string(),
                    Expression::Binary(
                        BinaryOperator::Add,
                        Box::new(Expression::Identifier("sum".to_string())),
                        Box::new(Expression::Identifier("i".to_string())),
                    )
                ),
            ]
        ),
        Statement::Expression(
            Expression::Call("println".to_string(), vec![
                Expression::Identifier("sum".to_string())
            ])
        ),
    ];

    let mut context = Context::create();

    let mut codegen = CodeGen::new(&mut context);

    codegen.generate(&program);
}
