import React, { Component } from 'react';
import axios from 'axios';

import { Form, Icon, Input, Button, Upload } from 'antd';
import { FormProps, FormComponentProps } from 'antd/lib/form';

function hasErrors(fieldsError: any) {
    return Object.keys(fieldsError).some(field => fieldsError[field]);
}

class DatasetForm extends React.Component<FormComponentProps> {
    componentDidMount() {
        // To disabled submit button at the beginning.
        this.props.form.validateFields();
    }

    handleSubmit = (e: any) => {
        e.preventDefault();
        this.props.form.validateFields((err, values) => {
            if (!err) {
                console.log('Received values of form: ', values);
                values['userId'] = 1;
                // values['file'] = values['files'][0]['originFileObj'];

                // delete values['fileList'];
                axios
                    .post(`http://127.0.0.1:5000/datasets`, values)
                    .then(res => {
                        // const dataset_items = res.data.map((data: any) => {
                        //     data['key'] = data['id'];
                        //     return data;
                        // });
                    });
            }
        });
    };

    normFile = (e: any) => {
        if (Array.isArray(e)) {
            return e;
        }
        return e && e.fileList;
    };

    render() {
        const {
            getFieldDecorator,
            getFieldsError,
            getFieldError,
            isFieldTouched
        } = this.props.form;

        // Only show error after a field is touched.
        const nameError = isFieldTouched('name') && getFieldError('name');
        const fileError =
            isFieldTouched('fileSelect') && getFieldError('fileSelect');
        return (
            <div style={{ padding: 10, borderRadius: 4 }}>
                <Form layout="vertical" onSubmit={this.handleSubmit}>
                    <Form.Item
                        validateStatus={nameError ? 'error' : ''}
                        help={nameError || ''}
                    >
                        {getFieldDecorator('datasetName', {
                            valuePropName: 'name',
                            rules: [
                                {
                                    required: true,
                                    message: 'Please select a dataset name.'
                                }
                            ]
                        })(
                            <Input
                                prefix={
                                    <Icon
                                        type="user"
                                        style={{ color: 'rgba(0,0,0,.25)' }}
                                    />
                                }
                                placeholder="Username"
                            />
                        )}
                    </Form.Item>
                    <Form.Item
                        validateStatus={fileError ? 'error' : ''}
                        help={fileError || ''}
                    >
                        {getFieldDecorator('files', {
                            valuePropName: 'fileSelect',
                            getValueFromEvent: this.normFile,
                            rules: [
                                {
                                    required: true,
                                    message: 'Please select a file.'
                                }
                            ]
                        })(
                            // <Input type="file" />
                            <Upload name="logo" listType="text" accept=".csv">
                                <Button>
                                    <Icon type="upload" /> Select file
                                </Button>
                            </Upload>
                        )}
                    </Form.Item>
                    <Form.Item>
                        <Button
                            type="primary"
                            htmlType="submit"
                            disabled={hasErrors(getFieldsError())}
                        >
                            Upload
                        </Button>
                    </Form.Item>
                </Form>
            </div>
        );
    }
}

const WrappedDatasetForm = Form.create({ name: 'dataset_form' })(DatasetForm);

// ReactDOM.render(<WrappedHorizontalLoginForm />, mountNode);
export default WrappedDatasetForm;
