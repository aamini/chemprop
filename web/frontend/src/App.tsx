import React, { Component } from 'react';
import './App.css';

import { Layout } from 'antd';
import NavBar from './components/headerbar/NavBar';
import HomePage from './components/pages/HomePage';

import { pageName } from './api/constants';

export interface AppState {
    currentPage: pageName;
}

export class App extends Component<{}, AppState> {
    readonly state: AppState = {
        currentPage: pageName.Home
    };

    changePage(newPage: pageName) {
        return () => this.setState({ currentPage: newPage });
    }

    render() {
        return (
            <div>
                <Layout style={{ backgroundColor: '#fff' }}>
                    <NavBar changePage={this.changePage} />
                    <HomePage currentPage={this.state.currentPage} />
                </Layout>
            </div>
        );
    }
}

export default App;
