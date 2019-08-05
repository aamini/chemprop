import React, { Component } from 'react';
import axios from 'axios';

import { Alert, Divider, Typography, Layout, Row, Col, Icon, Tag } from 'antd';

import { pageName } from '../../api/constants';
import { DataTable } from '../table/DataTable';
import WrappedDatasetForm from '../datasets/DatasetForm';

const { Content } = Layout;
const { Title, Paragraph } = Typography;

export interface HomePageProps {
	currentPage: pageName;
}

export interface HomePageState {
	dataset_items: {}[] | undefined;
}

export class HomePage extends Component<HomePageProps, HomePageState> {
	state = {
		dataset_items: undefined
	};

	componentDidMount() {
		this.updateDatasets();
	}

	updateDatasets() {
		axios.get(`http://127.0.0.1:5000/datasets`).then((res) => {
			const dataset_items = res.data.map((data: any) => {
				data['key'] = data['id'];
				return data;
			});

			this.setState({ dataset_items: dataset_items });
		});
	}

	deleteDataset(id: number) {
		return () =>
			axios.delete(`http://127.0.0.1:5000/datasets/${id}`).then((res) => {
				this.updateDatasets();
			});
	}

	render() {
		const dataSource = undefined;

		const dataset_columns = [
			{
				title: 'Name',
				dataIndex: 'datasetName',
				key: 'name',
				render: (name: string) => <b>{name}</b>
			},
			{
				title: 'Class',
				dataIndex: 'class',
				key: 'type',
				render: (type: string) => (
					<span>
						<Tag color={type === 'classification' ? 'red' : 'blue'}>{type.toUpperCase()}</Tag>
					</span>
				)
			},
			{
				title: 'Creation Time',
				dataIndex: 'created',
				key: 'created'
			},
			{
				title: 'Size',
				dataIndex: 'size',
				key: 'size'
			},
			{
				title: 'Action',
				key: 'action',
				render: (text: any, record: any) => (
					<span>
						<a href="javascript:;" onClick={this.deleteDataset(record.key)}>
							Delete
						</a>
					</span>
				)
			}
		];

		return (
			<Content
				style={{
					display: this.props.currentPage == pageName.Home ? 'block' : 'none',
					width: '100%',
					float: 'none',
					margin: 'auto',
					marginTop: 20,
					padding: 20
				}}
			>
				<Row gutter={32}>
					<Col className="gutter-row" span={8}>
						<DataTable dataSource={this.state.dataset_items} columns={dataset_columns} title="Datasets" />
						<WrappedDatasetForm />
					</Col>
					<Col className="gutter-row" span={8}>
						<DataTable dataSource={dataSource} columns={dataset_columns} title="Checkpoints" />
					</Col>
					<Col className="gutter-row" span={8}>
						<DataTable dataSource={dataSource} columns={dataset_columns} title="Predictions" />
					</Col>
				</Row>
			</Content>
		);
	}
}

export default HomePage;
