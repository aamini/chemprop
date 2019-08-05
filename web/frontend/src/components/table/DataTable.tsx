import React, { Component } from 'react';

import { Table } from 'antd';
import { ColumnProps } from 'antd/lib/table';

export interface TableProps {
	title: string;
	columns: ColumnProps<{}>[] | undefined;
	dataSource: {}[] | undefined;
}

export class DataTable extends Component<TableProps> {
	render() {
		return (
			<Table
				dataSource={this.props.dataSource}
				columns={this.props.columns}
				bordered
				size="small"
				title={() => <div style={{ fontWeight: 800 }}>{this.props.title}</div>}
			/>
		);
	}
}

export default DataTable;
