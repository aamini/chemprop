import React, { Component } from 'react';

import {
    Alert,
    Divider,
    Typography,
    Layout,
    Row,
    Col,
    Icon,
    Table
} from 'antd';

import { page, pageName } from '../../api/constants';

const { Content } = Layout;
const { Title, Paragraph } = Typography;

export interface HomePageProps {
    currentPage: pageName;
}

export class HomePage extends Component<HomePageProps> {
    render() {
        const dataSource = [
            {
                key: '1',
                name: 'Mike',
                age: 32,
                address: '10 Downing Street'
            },
            {
                key: '2',
                name: 'John',
                age: 42,
                address: '10 Downing Street'
            }
        ];

        const columns = [
            {
                title: 'Name',
                dataIndex: 'name',
                key: 'name'
            },
            {
                title: 'Age',
                dataIndex: 'age',
                key: 'age'
            },
            {
                title: 'Address',
                dataIndex: 'address',
                key: 'address'
            }
        ];
        return (
            <Content
                style={{
                    display:
                        this.props.currentPage == pageName.Home
                            ? 'block'
                            : 'none',
                    width: '100%',
                    float: 'none',
                    margin: 'auto',
                    marginTop: 20,
                    padding: 20
                }}
            >
                <Row gutter={32}>
                    <Col className="gutter-row" span={8}>
                        <Table
                            dataSource={dataSource}
                            columns={columns}
                            bordered
                            size="small"
                            title={() => (
                                <div style={{ fontWeight: 800 }}>Data</div>
                            )}
                        />
                    </Col>
                    <Col className="gutter-row" span={8}>
                        <Table
                            dataSource={dataSource}
                            columns={columns}
                            bordered
                            size="small"
                            title={() => (
                                <div style={{ fontWeight: 800 }}>Checkpoints</div>
                            )}
                        />
                    </Col>
                    <Col className="gutter-row" span={8}>
                        <Table
                            dataSource={dataSource}
                            columns={columns}
                            bordered
                            size="small"
                            title={() => (
                                <div style={{ fontWeight: 800 }}>Predictions</div>
                            )}
                        />
                    </Col>
                </Row>
            </Content>
        );
    }
}

export default HomePage;
