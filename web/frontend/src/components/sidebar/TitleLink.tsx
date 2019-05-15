import React, { Component } from 'react';

import { Typography } from 'antd';

const { Title } = Typography;

export class TitleLink extends Component {
    render() {
        return (
            <Title level={4} style={{display: 'inline'}}>
                ChemProp
            </Title>
        );
    }
}

export default TitleLink;
