import React, { useEffect, useState } from 'react';
import Papa from 'papaparse';
import { useTable } from 'react-table';
const predictionsCsv = process.env.PUBLIC_URL + '/predictions.csv';

function PredictionsTable() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch(predictionsCsv)
      .then((res) => res.text())
      .then((text) => {
        const parsed = Papa.parse(text, { header: true });
        setData(parsed.data);
      });
  }, []);

  const columns = React.useMemo(() => {
    if (data.length === 0) return [];
    return Object.keys(data[0]).map((key) => ({ Header: key, accessor: key }));
  }, [data]);

  const tableInstance = useTable({ columns, data });
  const { getTableProps, getTableBodyProps, headerGroups, rows, prepareRow } = tableInstance;

  return (
    <div>
      <h2>Predictions</h2>
      <table {...getTableProps()} style={{ border: '1px solid gray' }}>
        <thead>
          {headerGroups.map((headerGroup) => (
            <tr {...headerGroup.getHeaderGroupProps()}>
              {headerGroup.headers.map((column) => (
                <th {...column.getHeaderProps()}>{column.render('Header')}</th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody {...getTableBodyProps()}>
          {rows.map((row) => {
            prepareRow(row);
            return (
              <tr {...row.getRowProps()}>
                {row.cells.map((cell) => (
                  <td {...cell.getCellProps()}>{cell.render('Cell')}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default PredictionsTable;