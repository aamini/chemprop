import React, { Component } from 'react';
import './App.css';

import { Layout } from 'antd';
import NavBar from './components/sidebar/NavBar';

const { Content } = Layout;

export class App extends Component {
    render() {
        return (
            <div>
                <Layout>
                    <NavBar />
                    <Content></Content>
                </Layout>
            </div>
        );
    }
}

export default App;
