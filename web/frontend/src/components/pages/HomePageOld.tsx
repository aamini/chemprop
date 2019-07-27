import React, { Component } from 'react';

import { Alert, Divider, Typography, Layout, Row, Col, Icon } from 'antd';

import { page, pageName } from '../../api/constants';

const { Content } = Layout;
const { Title, Paragraph } = Typography;

export interface HomePageProps {
    currentPage: pageName;
}

export class HomePage extends Component<HomePageProps> {
    render() {
        return (
            <Content
                style={{
                    display: this.props.currentPage == pageName.Home ? 'block' : 'none',
                    maxWidth: page.width,
                    width: '100%',
                    float: 'none',
                    margin: 'auto',
                    marginTop: 40
                }}
            >
                <Row style={{ width: '100%' }} gutter={60}>
                    <Col span={12} style={{ textAlign: 'center' }}>
                        <img
                            src="/images/message_passing.png"
                            style={{ width: '80%' }}
                        />
                    </Col>
                    <Col
                        span={12}
                        style={{
                            textAlign: 'center',
                            marginTop: 50
                        }}
                    >
                        <Title>ChemProp</Title>
                        <Paragraph>
                            This website can be used to predict molecular
                            properties using a Message Passing Neural Network
                            (MPNN).
                        </Paragraph>
                    </Col>
                </Row>
                <Alert
                    message="In order to make predictions, an MPNN first needs to be trained on a dataset containing molecules along with known property values for each molecule. Once the MPNN is trained, it can be used to predict those same properties on any new molecules."
                    type="success"
                    style={{ marginTop: 30 }}
                />
                <Row style={{ width: '100%', marginTop: 30 }} gutter={60}>
                    <Col span={12}>
                        <Title level={2}>
                            <Icon type="database" style={{ marginRight: 20 }} />
                            Datasets
                        </Title>
                        <Paragraph>
                            View information about all datasets and upload more.
                        </Paragraph>
                    </Col>
                    <Col span={12}>
                        <Title level={2}>
                            <Icon type="flag" style={{ marginRight: 20 }} />
                            Checkpoints
                        </Title>
                        <Paragraph>
                            View information about all checkpoints and upload
                            more.
                        </Paragraph>
                    </Col>
                </Row>
                <Row style={{ width: '100%', marginTop: 50 }} gutter={60}>
                    <Col span={12}>
                        <Title level={2}>
                            <Icon type="rise" style={{ marginRight: 20 }} />
                            Train
                        </Title>
                        <Paragraph>
                            To train an MPNN, go to the Train page, upload a
                            dataset or select a dataset which has already been
                            uploaded, set the desired parameters, name the
                            model, and then click "Train".
                        </Paragraph>
                    </Col>
                    <Col span={12}>
                        <Title level={2}>
                            <Icon type="eye" style={{ marginRight: 20 }} />
                            Predict
                        </Title>
                        <Paragraph>
                            To make property predictions using ChemProp, go to
                            the Predict page, select the trained model
                            checkpoint you want to use, upload or paste the
                            molecules you would like to make predictions on, and
                            then click "Predict".
                        </Paragraph>
                    </Col>
                </Row>
                <Divider>Advanced Features</Divider>
                <Title level={4}>Using GPUs</Title>
                <Paragraph>
                    If GPUs are available on your machine, you will see a
                    dropdown menu on the Train and Predict pages which will
                    allow you to select a GPU to use. If you select "None" then
                    only CPUs will be used.
                </Paragraph>
                <Title level={4}>Working Remotely</Title>
                <Paragraph>
                    If you wish to train or predict on a remote server, you can
                    use SSH port-forwarding to run training/predicting on the
                    remote server while viewing the website it locally. To do
                    so, follow these instructions:
                </Paragraph>

                <ol>
                    <li>
                        Connect to the remote server:{' '}
                        <code>ssh &lt;remote_user&gt;@&lt;remote_host&gt;</code>
                    </li>
                    <li>On the remote server:</li>
                    <ol>
                        <li>
                            Navigate to the <code>chemprop</code> directory.
                        </li>
                        <li>
                            Activate the conda environment with the{' '}
                            <code>chemprop</code> requirements:{' '}
                            <code>
                                source activate &lt;environment_name&gt;
                            </code>
                        </li>
                        <li>
                            Start the website: <code>python web/run.py</code>
                        </li>
                    </ol>
                    <li>On your local machine:</li>
                    <ol>
                        <li>
                            Set up port-forwarding:{' '}
                            <code>
                                ssh -N -L 5000:localhost:5000
                                &lt;remote_user&gt;@&lt;remote_host&gt;
                            </code>
                        </li>
                        <li>
                            Open a web browser and go to{' '}
                            <a href="localhost:5000">localhost:5000</a>
                        </li>
                    </ol>
                </ol>
            </Content>
        );
    }
}

export default HomePage;
